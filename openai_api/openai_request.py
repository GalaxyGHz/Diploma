
def create_request(identifier="custom_id", model="gpt-4o", system_promt="Say 'You forgot to provide a system prompt!'", user_message="Hi!", logprobs=False, max_tokens=512):
    request = {
        "custom_id": identifier, 
        "method": "POST", 
        "url": "/v1/chat/completions", 
        "body": 
        {
            "model": model, 
            "messages": 
            [
                {"role": "system", "content": system_promt},
                {"role": "user", "content": user_message}
            ],
            "logprobs": logprobs,
            "max_tokens": max_tokens
        }
    }
    return request