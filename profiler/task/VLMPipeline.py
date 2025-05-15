import torch
import time
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from transformers.cache_utils import HybridCache

torch._dynamo.config.disable = True

class VLMPipeline:
    MODEL_NAME = "google/gemma-3-4b-it-qat-q4_0-unquantized"
    IMAGE_PATH = "../assets/test.jpg"
    OUTPUT_TOKENS = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self):
        self.model = None
        self.processor = None
        self.inputs = None
        self.input_ids = None
        self.attention_mask = None
        self.pixel_values = None
        self.token_type_ids = None
        self.prefill_seq_len = None
        self.batch_size = None
        self.past_kv = None
        self.prefill_output = None

    def load_model(self):
        print("\nLoading model...")
        start = time.time()
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(f"Model load time: {round(time.time() - start, 2)} seconds")

    def load_processor(self):
        print("\nLoading processor...")
        start = time.time()
        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_NAME,
            padding_side="left"
        )
        print(f"Processor load time: {round(time.time() - start, 2)} seconds")

    def get_messages(self):
        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": self.IMAGE_PATH},
                    {"type": "text", "text": "Describe the image in detail."},
                ]
            },
        ]

    def load_inputs(self, messages):
        print("\nLoading inputs...")
        start = time.time()
        self.inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.DEVICE)
        print(f"Input load time: {round(time.time() - start, 2)} seconds")

    def setup_cache(self):
        print("\nSetting up cache...")
        self.input_ids = self.inputs["input_ids"]
        self.attention_mask = self.inputs["attention_mask"]
        self.pixel_values = self.inputs["pixel_values"]
        self.token_type_ids = self.inputs.get("token_type_ids")

        shape = self.input_ids.shape
        self.batch_size, self.prefill_seq_len = shape[0], shape[1]
        print(f"Prefill sequence length: {self.prefill_seq_len}")

        total_seq_len = self.prefill_seq_len + self.OUTPUT_TOKENS
        print(f"Initializing cache for total sequence length: {total_seq_len}")

        cache_kwargs = {
            "max_batch_size": self.batch_size,
            "max_cache_len": total_seq_len
        }
        self.past_kv = HybridCache(
            self.model.language_model.config,
            device=self.DEVICE,
            dtype=self.model.dtype,
            **cache_kwargs
        )
        print("Created HybridCache instance.")

    def prefill(self):
        print("\nPrefilling...")
        start = time.time()
        inputs = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "pixel_values": self.pixel_values,
            "token_type_ids": self.token_type_ids,
            "past_key_values": self.past_kv,
            "use_cache": True,
            "output_attentions": False,
            "output_hidden_states": False,
        }
        with torch.no_grad():
            self.prefill_output = self.model(**inputs)
        self.past_kv = self.prefill_output.past_key_values
        print(f"Prefill time: {round(time.time() - start, 2)} seconds")

    def decode(self):
        print("\nDecoding...")
        start = time.time()
        eos_id = self.processor.tokenizer.eos_token_id

        next_logits = self.prefill_output.logits[:, -1, :]
        next_token = torch.argmax(next_logits, dim=-1)

        generated_ids = []
        current_mask = self.attention_mask

        for step in range(self.OUTPUT_TOKENS):
            inp_id = next_token.unsqueeze(-1)
            cache_pos = torch.tensor([self.prefill_seq_len + step], device=self.DEVICE)

            print(f"\n--- Step {step+1}/{self.OUTPUT_TOKENS} ---")
            tid = next_token.item()
            tstr = self.processor.decode([tid], skip_special_tokens=True)
            print(f"  Token ID: {tid} ('{tstr}')")
            print(f"  Cache Position: {cache_pos.item()}")

            with torch.no_grad():
                out = self.model(
                    input_ids=inp_id,
                    attention_mask=current_mask,
                    past_key_values=self.past_kv,
                    cache_position=cache_pos,
                    pixel_values=None,
                    token_type_ids=None,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )

            generated_ids.append(tid)
            if tid == eos_id:
                print("  EOS token encountered. Stopping.")
                break

            logits = out.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1)
            self.past_kv = out.past_key_values

            current_mask = torch.cat([
                current_mask,
                torch.ones((self.batch_size, 1), dtype=torch.long, device=self.DEVICE)
            ], dim=-1)

        result = self.processor.decode(generated_ids, skip_special_tokens=True)
        print("\n--- Decoding Finished ---")
        print("Result:", result)
        print(f"Decoding time: {round(time.time() - start, 2)} seconds")
        return result


if __name__ == "__main__":
    pipeline = VLMPipeline()
    pipeline.load_model()
    pipeline.load_processor()
    msgs = pipeline.get_messages()
    pipeline.load_inputs(msgs)
    pipeline.setup_cache()
    pipeline.prefill()
    pipeline.decode()