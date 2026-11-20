"""Offline textual inference with a local Hugging Face checkpoint."""
from pathlib import Path

from .engine import Inference


class OfflineNLI:
    def __init__(self, model_path: str, device: str = "cpu", max_length: int = 384):
        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(f"NLI model directory not found: {path}")

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False, dtype=torch.float32,
        ).to(device).eval()
        self.device = device
        self.max_length = max_length
        self.labels = {
            str(label).casefold().replace("-", "_").replace(" ", "_"): int(index)
            for index, label in self.model.config.id2label.items()
        }
        label_names = set(self.labels)
        self.binary = label_names == {"entailment", "not_entailment"}
        if not self.binary and label_names != {"entailment", "contradiction", "neutral"}:
            raise ValueError(f"Unsupported NLI labels: {self.labels}")

    def _probabilities(self, pairs, batch_size):
        results = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            encoded = self.tokenizer(
                [premise for premise, _ in batch], [hypothesis for _, hypothesis in batch],
                padding=True, truncation="only_first", max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.inference_mode():
                scores = self.model(**encoded).logits.float().softmax(dim=-1)
                results.extend(scores.cpu().tolist())
        return results

    def predict_batch(self, pairs: list[tuple[str, str]], batch_size: int = 12) -> list[Inference]:
        entailment_index = self.labels["entailment"]
        if not self.binary:
            return [
                Inference(row[entailment_index], row[self.labels["contradiction"]],
                          row[self.labels["neutral"]])
                for row in self._probabilities(pairs, batch_size)
            ]

        # The catalogue model pools neutral and contradiction into not_entailment.
        # A contradiction needs affirmative support for the explicit denial.
        negated_pairs = [
            (premise, self._denial(hypothesis))
            for premise, hypothesis in pairs
        ]
        scores = self._probabilities(pairs + negated_pairs, batch_size)
        result = []
        for assertion, denial in zip(scores[:len(pairs)], scores[len(pairs):]):
            positive = assertion[entailment_index]
            negative = denial[entailment_index]
            entailment = min(positive, 1 - negative)
            contradiction = min(negative, 1 - positive)
            result.append(Inference(entailment, contradiction, max(0.0, 1 - entailment - contradiction)))
        return result

    def passing_support(self, pairs, threshold, batch_size=12):
        """Return exact entailment for passing pairs and zero for the rest."""
        if not pairs:
            return []
        index = self.labels["entailment"]
        positive = [row[index] for row in self._probabilities(pairs, batch_size)]
        passing = [i for i, score in enumerate(positive) if score >= threshold]
        result = [0.0] * len(pairs)
        if self.binary and passing:
            denials = [(pairs[i][0], self._denial(pairs[i][1])) for i in passing]
            for i, row in zip(passing, self._probabilities(denials, batch_size)):
                score = min(positive[i], 1 - row[index])
                result[i] = score if score >= threshold else 0.0
        else:
            for i in passing:
                result[i] = positive[i]
        return result

    @staticmethod
    def _denial(hypothesis):
        return f"It is false that {hypothesis.strip().rstrip('.')}."

    def predict(self, premise: str, hypothesis: str) -> Inference:
        return self.predict_batch([(premise, hypothesis)])[0]
