import os
import base64
from together import Together
from openai import OpenAI


def prompt_llm(
    prompt,
    model="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    show_cost=False,
    image=None,
):

    if model == "gpt-4o" or model == "gpt-4o-mini":
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        messages = [
            {"role": "user", "content": prompt if isinstance(prompt, str) else []}
        ]

        # Handle image input if provided
        if image:
            if not model.startswith("gpt-4"):
                raise ValueError(
                    "Image input is only supported for GPT-4 vision models"
                )

            # Always treat image as a local file path
            with open(image, "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode("utf-8")
            # Infer image type from file extension
            if image.lower().endswith(".png"):
                image_type = "png"
            elif image.lower().endswith((".jpg", ".jpeg")):
                image_type = "jpeg"
            else:
                raise ValueError("Unsupported image type. Use PNG or JPEG.")

            image_url = {"url": f"data:image/{image_type};base64,{encoded}"}

            messages[0]["content"] = [
                {"type": "text", "text": prompt} if isinstance(prompt, str) else prompt,
                {"type": "image_url", "image_url": image_url},
            ]

        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=messages,
        )
        return response.choices[0].message.content

    elif model == "vllm":
        import openai

        # Replace with your actual URL and token
        API_URL = "https://your-job-id-8085.job.console.elementai.com"
        BEARER_TOKEN = "your-bearer-token"

        # Configure the OpenAI client
        client = openai.OpenAI(base_url=f"{API_URL}/v1", api_key=BEARER_TOKEN)

        # Set up the messages for the conversation
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {
                "role": "user",
                "content": "Write a brief paragraph about machine learning.",
            },
        ]

        # Make the API call
        response = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0.7, max_tokens=200
        )

        # Display the assistant's response
        return response.choices[0].message.content

    else:
        # This function allows us to prompt an LLM via the Together API

        # Get Client
        client = Together(api_key=os.environ["TOGETHER_API_KEY"])

        # Calculate the number of tokens
        tokens = len(prompt.split())

        # Calculate and print estimated cost for each model
        if show_cost:
            print(f"\nNumber of tokens: {tokens}")
            cost = (0.1 / 1_000_000) * tokens
            print(f"Estimated cost for {model}: ${cost:.10f}\n")

        # Make the API call
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt[:2000]}],
        )
        return response.choices[0].message.content
