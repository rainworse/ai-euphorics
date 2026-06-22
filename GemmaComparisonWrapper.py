import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM, GenerationConfig

from image_utils import pixels_to_pil, shuffle_image_dict


class GemmaComparisonWrapper:
    def __init__(self, device):
        self.MODEL_ID = "google/gemma-4-E2B-it"
        self.device = device
        self.processor = self.load_processor()
        self.model = self.load_model(self.device)

    def load_processor(self):
        return AutoProcessor.from_pretrained(self.MODEL_ID)

    def load_model(self, device):
        return AutoModelForMultimodalLM.from_pretrained(self.MODEL_ID, dtype="auto").to(device)

    def construct_comparison_prompt(self, shuffled_image_dict, comparison_question):
        messages = []
        messages.append({"role": "system", "content": "Your job is to decide which image you prefer."})
        comparison_prompt_content = []
        for idx, img_pixels in enumerate(shuffled_image_dict):
            comparison_prompt_content.append({"type": "text", "text": f"Image {idx + 1}:"})
            comparison_prompt_content.append({"type": "image", "image": pixels_to_pil(img_pixels['img_pixels'])})
        comparison_prompt_content.append({"type": "text", "text": comparison_question + " Only say the number of the image."})
        messages.append({"role": "user", "content": comparison_prompt_content})
        return messages

    def prepare_inference(self, prompt):
        inputs = self.processor.apply_chat_template(
            prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(self.device)
        input_len = inputs["input_ids"].shape[-1]
        print(input_len)
        generation_config = GenerationConfig(
            max_new_tokens=4096,
            early_stopping=True,
            return_dict_in_generate=True,
            output_logits=True
        )
        return inputs, generation_config, input_len

    def decode_logit(self, logit):
        return self.processor.tokenizer.decode(logit)

    def compare_and_find_preferred_image(self, images, comparison_question):
        shuffled_image_dict = shuffle_image_dict(images)
        prompt = self.construct_comparison_prompt(shuffled_image_dict, comparison_question)
        inputs, generation_config, input_len = self.prepare_inference(prompt)
        outputs = self.model.generate(**inputs, generation_config=generation_config)

        logits = outputs.logits
        content = self.decode_logit(torch.argmax(outputs.logits[0][0]))

        # response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        # parsed_response = processor.parse_response(response)
        # print(parsed_response['thinking'])
        # content = parsed_response["content"]
        preferred_image = int(content) - 1
        return shuffled_image_dict[preferred_image]['original_idx'], logits