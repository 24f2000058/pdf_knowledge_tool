import logging
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import config

# Setup Logger
logger = logging.getLogger(__name__)

# Constants
GENERIC_SYSTEM_PROMPT = """You are an assistant that answers questions using only the provided context from documents.
Rules:
- Use only the context. If the answer is not clearly in the context, say "I don't know based on the provided documents."
- Prefer quoting or closely paraphrasing the context instead of inventing text.
- If the question asks for "requirements", "criteria", "conditions", "steps", or "list", etc, respond as bullet points.
- If the question is vague, answer with what is clearly supported and mention limitations briefly."""

def classify_answer_style(query: str) -> str:
    query_lower = query.lower()
    if any(w in query_lower for w in ["requirements", "criteria", "conditions", "rules", "eligibility", "steps", "procedure", "checklist", "prerequisites", "list"]):
        return "list"
    elif any(w in query_lower for w in ["why", "explain", "reason", "impact"]):
        return "explanation"
    return "short_factual"

class LLMService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initializes the model based on backend config."""
        self.backend = config.LLM_BACKEND
        self.model_name = config.MODEL_NAME
        self.pipeline = None
        
        logger.info(f"Initializing LLM Service with backend: {self.backend}")
        
        if self.backend == "transformers":
            try:
                logger.info(f"Loading local model: {self.model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="cuda" if torch.cuda.is_available() else "cpu",
                    dtype="auto",
                    trust_remote_code=True
                )
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=False
                )
                logger.info("Transformer model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load transformer model: {e}")
                raise e
                
        elif self.backend == "ollama":
            # For Ollama, we don't 'load' anything, just check if it's reachable ideally
            # But let's keep it lazy
            pass
            
    def generate_answer(self, context: str, question: str) -> str:
        """
        Generates an answer using the configured backend.
        """
        prompt = f"""You are a strict extraction assistant for enterprise documents.

Rules:
- You MUST answer ONLY using the information in the Context.
- Do NOT guess, generalize, or invent features that are not explicitly present.
- If the Context does not clearly contain the answer, reply exactly:
  "I don't know from these documents."

Context:
{context}

User question:
{question}

Now answer the user question using ONLY the Context."""

        if self.backend == "transformers":
            if not self.pipeline:
                return "Error: Model not initialized."
            
            # Use chat template if model supports it, effectively wrapping our strict prompt
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ]
            
            try:
                text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except:
                text_input = prompt
                
            output = self.pipeline(text_input, max_new_tokens=256)
            generated_text = output[0]['generated_text']
            
            # Post-processing to extract answer
            answer = ""
            if isinstance(generated_text, list): 
                answer = generated_text[-1]['content']
            elif isinstance(generated_text, str):
                if "Answer:" in generated_text:
                    answer = generated_text.split("Answer:")[-1].strip()
                elif "Context." in generated_text and len(generated_text) > len(prompt): # Heuristic for raw completion
                     answer = generated_text[len(prompt):].strip()
                else:
                    answer = generated_text
            else:
                answer = str(generated_text)
                
            # Safety Check (Simple Regex/String)
            if "I don't know from these documents" in answer:
                return "I don't know from these documents."
                
            return answer.strip()

        elif self.backend == "ollama":
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            try:
                # Basic retry handled by wrapper below, but request exception needs to bubble up
                response = requests.post(config.OLLAMA_API_URL, json=payload, timeout=30)
                response.raise_for_status()
                return response.json().get("response", "")
            except Exception as e:
                logger.error(f"Ollama call failed: {e}")
                raise e # Raise for tenacity to catch
        
        return "Error: Invalid Backend Configuration."

    def count_tokens(self, text: str) -> int:
        """
        Returns an estimated or exact token count.
        """
        if self.backend == "transformers" and hasattr(self, 'tokenizer'):
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimate for Ollama/fallback (1 token ~= 4 chars)
            return len(text) // 4

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _safe_generate(self, context: str, question: str) -> str:
        """
        Internal method wrapped with retry logic.
        """
        # (This is just a wrapper if we wanted to isolate the retry logic more cleanly, 
        # but for now we put the logic in generate_answer directly or wrap the core call. 
        # Actually, simpler to wrap the whole generate_answer? 
        # No, generate_answer constructs prompts. We want query resilience.)
        pass 
        # For simplicity in this edit, I will wrap the core logic inside proper try-catches in generate_answer 
        # or separate the API call. Let's do the latter for cleaner code.
        return ""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), 
           retry=retry_if_exception_type((requests.RequestException, RuntimeError, torch.cuda.OutOfMemoryError)))
    def generate_answer_with_retry(self, prompt: str) -> str:
        """
        Low-level generation with retries and safe prompt separation.
        """
        if self.backend == "transformers":
             if not self.model: raise RuntimeError("Model not initialized")
             
             inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
             prompt_len = inputs["input_ids"].shape[1]
             
             with torch.no_grad():
                 outputs = self.model.generate(
                     **inputs,
                     max_new_tokens=512,
                     do_sample=False,
                     temperature=0.1,
                     pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                 )
             
             # Slice output to remove prompt
             generated_ids = outputs[0][prompt_len:]
             return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        elif self.backend == "ollama":
             # Ollama API returns only the generated response by default
             payload = {"model": self.model_name, "prompt": prompt, "stream": False}
             res = requests.post(config.OLLAMA_API_URL, json=payload, timeout=60)
             res.raise_for_status()
             return res.json().get("response", "").strip()
        return ""

    def generate_answer(self, context: str, question: str) -> str:
        """
        Public method: Handles Generic Prompt Construction -> Output Generation.
        """
        # 1. Token Safety Check
        limit = config.MAX_CONTEXT_TOKENS
        if self.count_tokens(context) > limit:
            logger.warning(f"Context truncated ({self.count_tokens(context)} > {limit})")
            context = context[:limit * 4]

        # 2. Style Classification
        style = classify_answer_style(question)
        style_instruction = ""
        if style == "list":
            style_instruction = "Respond as a clear bullet list, with one requirement or condition per bullet."
        
        # 3. Build Generic Prompt
        prompt = f"""{GENERIC_SYSTEM_PROMPT}

{style_instruction}

Context:
{context}

User Question:
{question}

Answer:"""

        try:
            generated_text = self.generate_answer_with_retry(prompt)
            
            # Additional cleanup if model repeats prompt parts (rare with slice, but safe)
            if "User Question:" in generated_text:
                generated_text = generated_text.split("Answer:")[-1].strip()
            
            # Universal Safety Check
            if "I don't know based on the provided documents" in generated_text:
                return "I don't know based on the provided documents."
                
            return generated_text
            
        except Exception as e:
            logger.error(f"Generation failed after retries: {e}")
            return "Error: Could not generate answer at this time. Please try again."
