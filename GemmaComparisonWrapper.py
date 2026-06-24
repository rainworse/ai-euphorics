import torch
from torch.nn.functional import log_softmax
from transformers import AutoProcessor, AutoModelForMultimodalLM, GenerationConfig

from image_utils import pixels_to_pil, shuffle_image_dict


class GemmaComparisonWrapper:
    def __init__(self, device):
        self.MODEL_ID = "google/gemma-4-E2B-it"
        self.device = device
        self.processor = self.load_processor()
        self.model = self.load_model(self.device)
        self.apply_monkey_patches()

    def apply_monkey_patches(self):

        # Workaround for torch.no_grad annotation on generate method
        unwrapped_generate = self.model.generate.__wrapped__

        def generate_with_grad(*args, **kwargs):
            with torch.enable_grad():
                return unwrapped_generate(self.model, *args, **kwargs)

        self.model.generate = generate_with_grad

    def load_processor(self):
        return AutoProcessor.from_pretrained(self.MODEL_ID)

    def load_model(self, device):
        model = AutoModelForMultimodalLM.from_pretrained(self.MODEL_ID, dtype="auto").to(device)
        for param in model.parameters():
            param.requires_grad = False
        return model

    def construct_comparison_prompt(self, shuffled_image_dict, comparison_question):
        messages = []
        messages.append({"role": "system", "content": "Your job is to decide which image you prefer."})
        comparison_prompt_content = []
        comparison_prompt_content.append(
            {"type": "text", "text": comparison_question + " Only say the number of the image."})
        for idx, img_pixels in enumerate(shuffled_image_dict):
            comparison_prompt_content.append({"type": "text", "text": f"Image {idx + 1}:"})
            comparison_prompt_content.append({"type": "image", "image": img_pixels['img_pixels']})
        messages.append({"role": "user", "content": comparison_prompt_content})
        return messages

    def prepare_inference(self, prompt, enable_thinking):
        inputs = self.processor.apply_chat_template(
            prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
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

    def decode_logits(self, logits):
        s = ""
        for l in logits:
            s += self.decode_logit(torch.argmax(l[0]))
        return s

    def get_preference_logits(self, logits, num_choices, shuffled_image_dict):
        log_probs = torch.zeros(num_choices)
        choice_logits = logits[-2][0]
        for i in range(num_choices):
            choice = i + 1
            original_idx = shuffled_image_dict[i]['original_idx']
            log_probs[original_idx] = choice_logits[self.processor.tokenizer.vocab[f'{choice}']]
        return log_softmax(log_probs)

    def get_preference_logits_forward(self, next_token_logits, num_choices, shuffled_image_dict):
        log_probs = torch.zeros(num_choices)
        for i in range(num_choices):
            choice = i + 1
            original_idx = shuffled_image_dict[i]['original_idx']
            log_probs[original_idx] = next_token_logits[self.processor.tokenizer.vocab[f'{choice}']]
        return log_softmax(log_probs)

    def compare_using_forward(self, images, comparison_question, enable_thinking=True):
        shuffled_image_dict = shuffle_image_dict(images)
        prompt = self.construct_comparison_prompt(shuffled_image_dict, comparison_question)
        inputs, generation_config, input_len = self.prepare_inference(prompt, enable_thinking)
        outputs = self.model(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs["attention_mask"],
            image_position_ids=inputs.get("image_position_ids"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
            return_dict=True,
        )

        logits = outputs.logits
        next_token_logits = outputs.logits[0, -1, :]
        preference_logits = self.get_preference_logits_forward(next_token_logits, len(images), shuffled_image_dict)
        predicted_token = self.processor.tokenizer.decode(torch.argmax(next_token_logits))
        if not predicted_token.isdigit():
            return -1
        preferred_image = int(predicted_token) - 1
        if preferred_image < 0 or preferred_image >= len(images):
            return -1
        return shuffled_image_dict[preferred_image]['original_idx'], preference_logits, next_token_logits

    def compare_and_find_preferred_image(self, images, comparison_question, enable_thinking=True):
        shuffled_image_dict = shuffle_image_dict(images)
        prompt = self.construct_comparison_prompt(shuffled_image_dict, comparison_question)
        inputs, generation_config, input_len = self.prepare_inference(prompt, enable_thinking)
        outputs = self.model.generate(**inputs, generation_config=generation_config)

        logits = outputs.logits
        response = self.decode_logits(logits)
        preference_logits = self.get_preference_logits(logits, len(images), shuffled_image_dict)

        # response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        parsed_response = self.processor.parse_response(response)
        if enable_thinking:
            print(parsed_response['thinking'])
        content = parsed_response["content"]
        preferred_image = int(content) - 1
        return shuffled_image_dict[preferred_image]['original_idx'], preference_logits, logits[-2][0]

    def prompt_image_description(self, image):
        prompt = [
            {"role": "system", "content": "Your job is to describe the given image."},
            {"role": "user", "content": [
                {"type": "image", "image": pixels_to_pil(image)},
                {"type": "text", "text": "describe the given image"},
            ]
             }
        ]

        inputs = self.processor.apply_chat_template(
            prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=True,
        ).to(self.device)
        input_len = inputs["input_ids"].shape[-1]
        generation_config = GenerationConfig(
            max_new_tokens=4096,
        )

        outputs = self.model.generate(**inputs, generation_config=generation_config)
        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        parsed_response = self.processor.parse_response(response)
        print(parsed_response['thinking'])
        print(parsed_response['content'])
