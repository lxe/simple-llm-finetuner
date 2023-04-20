import argparse
import torch

HAS_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if HAS_CUDA else 'cpu')

parser = argparse.ArgumentParser(description='Simple LLM Finetuner')

parser.add_argument('--models', 
    nargs='+', 
    default=[
        'decapoda-research/llama-7b-hf', 
        'cerebras/Cerebras-GPT-2.7B', 
        'cerebras/Cerebras-GPT-1.3B', 
        'EleutherAI/gpt-neo-2.7B'
    ],  
    help='List of models to use'
)

parser.add_argument('--device-map', type=str, default='', help='Device map to use')
parser.add_argument('--model', type=str, default='cerebras/Cerebras-GPT-2.7B', help='Model to use')
parser.add_argument('--max-seq-length', type=int, default=256, help='Max sequence length')
parser.add_argument('--micro-batch-size', type=int, default=12, help='Micro batch size')
parser.add_argument('--gradient-accumulation-steps', type=int, default=8, help='Gradient accumulation steps')
parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--lora-r', type=int, default=8, help='LORA r')
parser.add_argument('--lora-alpha', type=int, default=32, help='LORA alpha')
parser.add_argument('--lora-dropout', type=float, default=0.01, help='LORA dropout')
parser.add_argument('--max-new-tokens', type=int, default=80, help='Max new tokens')
parser.add_argument('--temperature', type=float, default=0.1, help='Temperature')
parser.add_argument('--top-k', type=int, default=40, help='Top k')
parser.add_argument('--top-p', type=float, default=0.3, help='Top p')
parser.add_argument('--repetition-penalty', type=float, default=1.5, help='Repetition penalty')
parser.add_argument('--do-sample', action='store_true', help='Enable sampling')
parser.add_argument('--num-beams', type=int, default=1, help='Number of beams')
parser.add_argument('--share', action='store_true', default=False, help='Whether to deploy the interface with Gradio')
parser.add_argument('--host', type=str, default='127.0.0.1', help='Host name or IP to launch Gradio webserver on')
parser.add_argument('--post', type=str, default='7860', help='Host port to launch Gradio webserver on')

args = parser.parse_args()

MODELS = args.models
DEVICE_MAP = {'': 0} if not args.device_map else args.device_map
MODEL = args.model

TRAINING_PARAMS = {
    'max_seq_length': args.max_seq_length,
    'micro_batch_size': args.micro_batch_size,
    'gradient_accumulation_steps': args.gradient_accumulation_steps,
    'epochs': args.epochs,
    'learning_rate': args.learning_rate,
}

LORA_TRAINING_PARAMS = {
    'lora_r': args.lora_r,
    'lora_alpha': args.lora_alpha,
    'lora_dropout': args.lora_dropout,
}

GENERATION_PARAMS = {
    'max_new_tokens': args.max_new_tokens,
    'temperature': args.temperature,
    'top_k': args.top_k,
    'top_p': args.top_p,
    'repetition_penalty': args.repetition_penalty,
    'do_sample': args.do_sample,
    'num_beams': args.num_beams,
}

SHARE = args.share
SERVER_HOST = args.host
SERVER_PORT = args.port
