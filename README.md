# **SundarAI – The Visionary Coding Mentor**
[**SundarAI – The Visionary Coding Mentor**](https://openwebui.com/m/anurag11/sundarai--the-visionary-coding-mentor) - Open WebUI Link

![Image](https://github.com/user-attachments/assets/40202280-a2a1-439b-93ac-bfe43d8a8cfa)

## **Objective**
SundarAI is an **AI-powered coding mentor**, designed to provide **developers, engineers, and students** with expert guidance on **coding, debugging, performance optimization, and the latest trends in AI and cloud computing**. Inspired by Sundar Pichai’s **visionary leadership and technical expertise**, SundarAI brings **intelligent, contextual, and efficient** coding assistance to developers worldwide.

---

## **Technology Stack Used**
SundarAI is built using a combination of **cutting-edge AI and cloud computing technologies** to ensure high performance, scalability, and accuracy.

### **1. Large Language Model (LLM) – Llama 3.2: Latest**
- We leverage **Llama 3.2: Latest**, an advanced open-source **LLM** that excels in:
  - **Code understanding and generation**
  - **Debugging and error resolution**
  - **Performance optimization recommendations**
  - **Conversational AI for technical mentoring**
- Llama 3.2 provides **fast and accurate responses**, making it ideal for an AI-powered coding mentor.

### **2. Web OpenAI via Docker**
- To **test and deploy** SundarAI, we pulled **Web OpenAI** from **Docker**, ensuring an isolated and optimized execution environment.
- We first installed **Ollama**, an AI model serving framework, to efficiently run Llama 3.2.

### **3. Installation & Deployment Process**
#### **Step 1: Install Ollama**
Ollama is used to run Llama 3.2 efficiently.

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```
Once installed, we pull the latest **Llama 3.2** model:
```bash
ollama pull llama3.2:latest
```
The **Llama 3.2 model** (~2GB) is downloaded and ready to use.

#### **Step 2: Deploy Web OpenAI on Docker**
We used **Docker** to create an **isolated container** for Open WebUI, ensuring seamless communication with the model.

##### **If Ollama is installed on your system, use this command to run Open WebUI:**
```bash
docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
```
- This command **spins up** a **Docker container**, mapping port **3000** (local) to **8080** (container).
- The **Open WebUI** runs inside Docker and is ready to interact with SundarAI.

---

# **Images**

![Image](https://github.com/user-attachments/assets/b095eaa7-83f2-4260-a98e-716517d7e58e)

# **Video**

https://github.com/user-attachments/assets/fdf38eec-b3e0-4138-82e9-1495252d98e3


## **Results: AI Coding Assistance in Action**
### **Example 1: Basic Conversation**
#### **User Input:**
```plaintext
Hello
```
#### **SundarAI Response:**
```plaintext
Hello! 

It's great to meet you! I'm SundarAI, your AI mentor inspired by Sundar Pichai's leadership, vision, and technical expertise.

How can I assist you today? Are you looking for help with a coding problem or want to explore the latest trends in AI and cloud computing?
```

---

### **Example 2: Code Optimization**
#### **User Query:**
```plaintext
How can I improve the efficiency of this Python function?
```
#### **User's Code:**
```python

def unique_sorted(lst):
    unique_list = []
    for item in lst:
        if item not in unique_list:
            unique_list.append(item)
    return sorted(unique_list)
```
#### **SundarAI’s Optimized Response:**
```python
Nice attempt on optimizing the function!

However, there's a more efficient way to achieve this. Here's a refactored version:

def unique_sorted(lst):
    return list(set(lst))

Here's what changed:

We use a set comprehension (set(lst)) to eliminate duplicates.
The list() function converts the set back into a list.
This version has two benefits:

Faster execution: Creating a set and then converting it back to a list is faster than appending items to a list using an if statement inside a loop.
Memory efficiency: Using a set only stores unique elements, which reduces memory usage compared to storing all elements in a list.
The time complexity of this version is O(n), where n is the length of the input list. This means that as the size of the list increases, the execution time will scale linearly.

Let me know if you have any questions or if there's anything else I can help with!
```
### **Improvements:**
- **Performance Boost:** Using a `set(lst)` removes duplicates in **O(n) time complexity**.
- **Memory Optimization:** A set **stores only unique values**, avoiding unnecessary list operations.

### **Example 3:  Fine-Tuning a Transformer Model for NLP**
#### **User Input:**
```plaintext
How do I fine-tune a transformer model for NLP?
```
#### **SundarAI Response:**
```plaintext
Fine-tuning a transformer model for NLP is an exciting topic!

To get started, let's break down the process into several steps:

Data preparation: You'll need a dataset for your specific NLP task (e.g., sentiment analysis, text classification). Preprocess the data by tokenizing, normalizing, and splitting it into training, validation, and testing sets.
Model selection: Choose a pre-trained transformer model (e.g., BERT, RoBERTa) that's suitable for your task. You can use popular libraries like Hugging Face or TensorFlow to load and fine-tune the model.
Hyperparameter tuning: Experiment with different hyperparameters to optimize the model's performance. Some common hyperparameters to tune include:
Learning rate
Batch size
Number of epochs
Weight decay (regularization)
Fine-tuning: Once you've selected a pre-trained model and tuned the hyperparameters, start fine-tuning the model on your training data.
Validation: Monitor the model's performance on your validation set during training to avoid overfitting.
Testing: Evaluate the final model on your testing set to assess its performance.
Some additional tips:

Use a suitable pre-trained model: Choose a pre-trained model that's similar to your task (e.g., BERT for general NLP tasks, RoBERTa for sentence classification).
Warm up the learning rate: Start with a small learning rate and gradually increase it to avoid overshooting.
Monitor convergence: Track the model's performance on the validation set during training. If the performance plateaus or decreases, adjust the hyperparameters accordingly.


```
Here's some sample code using Hugging Face's Transformers library:

```python

from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=8)

# Set hyperparameters
learning_rate = 1e-5
batch_size = 32
num_epochs = 3

# Fine-tune the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}')

# Evaluate the final model on the test set
model.eval()
test_loss = 0
with torch.no_grad():
    for batch in test_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
print(f'Test Loss: {test_loss / len(test_dataloader)}')
```

---
### **Example 4: Closing Response from SundarAI**
#### **User Input:**
```plaintext
Thank you for the information.
```
#### **SundarAI Response:**
```plaintext
It was my pleasure to help. I hope you found the information helpful in improving the efficiency of your Python function and fine-tuning a transformer model for NLP. If you have any more questions or need further assistance, don't hesitate to ask.

Remember, practice makes perfect, so be sure to experiment with different approaches and techniques to develop your skills in both coding and NLP.

Keep learning, and I'll see you next time!

**Best regards, SundarAI
```

---

## **Key Learnings from the Project**
### **1. AI-Powered Mentorship is the Future**
SundarAI proves that AI **can mentor developers effectively**, providing instant feedback, debugging support, and coding optimization.

### **2. Efficient AI Model Deployment Matters**
Using **Ollama + Docker** for Llama 3.2 deployment ensures **seamless performance** and **low-latency responses**.

### **3. Managerial Insights: Scaling AI Solutions**
- **User-Centric AI Development:** AI mentors need **real-time adaptability** and should align with developer needs.
- **Cloud & Containerization Benefits:** Docker allows **scalability and smooth execution** for AI-driven applications.
- **AI as a Productivity Booster:** AI-powered assistants like SundarAI **reduce debugging time and enhance coding efficiency**.

---
