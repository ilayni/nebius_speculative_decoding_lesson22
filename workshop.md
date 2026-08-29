```python
from google.colab import userdata

from huggingface_hub import login
login(token=userdata.get('huggingface_api_key'))
```

# Task #1: Implement vanilla Speculative Decoding

Step 1: Generate from small model, with vectors of probabilities from big and small models (p(x) and q(x))

Step 2: Speculative sampling (accepance rule & last token sampling)

Step 3: Wrap in a loop and coloring

Step 4: Inference on Fineweb sample and measure acceptance rate and walltime

### ![Screenshot 2026-06-17 at 0.01.31.png]([IMAGE_STRIPPED])


```python
!pip install colorama
```


```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import colorama

device_type = 'cuda'

big_model_name =  "google/gemma-3-1b-pt"
small_model_name = "google/gemma-3-270m"

torch_dtype = torch.bfloat16
```


```python
input_text = 'once upon'
gamma = 5
seqlen = 30
```

Step 1: Generate from small model, with vectors of probabilities from big and small models (p(x) and q(x))

$q(x) \in R^{[\gamma, vocab]}$

$p(x) \in R^{[\gamma, vocab]}$


```python
## TODO ##
```
