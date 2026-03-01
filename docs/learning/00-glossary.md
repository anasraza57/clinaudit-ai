# Glossary of Terms

> A master reference for every technical term used in this project.
> Terms are defined in plain English with real-world analogies where helpful.

---

### API (Application Programming Interface)
A set of rules that allows one piece of software to talk to another. Like a waiter in a restaurant — you (the client) tell the waiter (API) what you want, the waiter goes to the kitchen (server), and brings back your food (response). You never go to the kitchen directly.

### Async / Asynchronous
A programming pattern where the program doesn't have to wait for a slow operation (like a database query or API call) to finish before doing other work. Like ordering food at a counter and sitting down — you don't stand there blocking the queue while they cook.

### CORS (Cross-Origin Resource Sharing)
A security feature in web browsers that controls which websites can make requests to your API. Without CORS configuration, a frontend running on `localhost:3000` can't talk to an API on `localhost:8000`.

### Cosine Similarity
A mathematical way to measure how similar two vectors (lists of numbers) are. It measures the angle between them: 1.0 means identical direction (very similar), 0.0 means perpendicular (unrelated), -1.0 means opposite. Used in this project to find guidelines most relevant to a search query.

### Docker
A tool that packages your application and all its dependencies into a standardised "container" that runs identically on any machine. Solves the "it works on my machine" problem.

### Docker Compose
A tool for defining and running multi-container Docker applications. Our `docker-compose.yml` starts both the application AND the database with a single command.

### Embedding
A way to convert text into a list of numbers (a "vector") that captures the meaning of the text. Texts with similar meanings produce similar vectors, which allows computers to understand semantic similarity.

### Endpoint
A specific URL path in an API that handles a particular type of request. For example, `/health` is our health check endpoint, and `/api/v1/audit` will be our audit endpoint.

### FAISS (Facebook AI Similarity Search)
A library by Meta for efficiently searching through large collections of vectors. Like a smart filing system that can instantly find the documents most similar to your query, even among millions.

### FastAPI
A modern Python web framework for building APIs. Chosen for its speed, automatic documentation, and built-in data validation.

### FHIR (Fast Healthcare Interoperability Resources)
A standard for exchanging healthcare information electronically. Like a universal translator for health IT systems.

### LLM (Large Language Model)
AI models trained on massive amounts of text that can understand and generate human language. Examples: GPT-4, Claude, Llama, Mistral.

### Migration (Database)
A version-controlled change to a database's structure. Like tracking every renovation you've ever made to a house — you can see exactly what changed and when, and can roll back if needed.

### NICE (National Institute for Health and Care Excellence)
A UK government body that publishes evidence-based guidelines on how healthcare conditions should be treated. The "answer key" our system checks against.

### ORM (Object-Relational Mapper)
A tool that lets you work with database records as Python objects instead of writing raw SQL. SQLAlchemy is our ORM.

### PostgreSQL
A powerful, open-source relational database. The industry standard for applications that need reliable, concurrent data storage.

### PubMedBERT
A version of BERT (a language understanding AI) that was trained specifically on millions of medical research papers. It understands medical terminology much better than general-purpose models.

### Pydantic
A Python library for data validation. It ensures that data matches expected types and formats — if you expect a number and get a string, it catches it immediately.

### RAG (Retrieval-Augmented Generation)
A technique where you first retrieve relevant documents from a database, then include them in the prompt to an LLM. This grounds the LLM's answers in real data instead of relying on its training memory.

### REST API
A style of API design where each URL represents a resource and you use HTTP methods (GET, POST, PUT, DELETE) to interact with them.

### SNOMED CT (Systematized Nomenclature of Medicine — Clinical Terms)
An international dictionary where every medical concept has a unique code. Ensures "low back pain", "lumbar pain", and "LBP" are all understood as the same thing by computers.

### SQLAlchemy
Python's most popular database toolkit and ORM. Lets us define database tables as Python classes and query them without writing raw SQL.

### Strategy Pattern
A design pattern where you define a family of algorithms (in our case, AI providers), make them interchangeable, and let the client choose which one to use at runtime. This is how we can swap between OpenAI, Anthropic, etc.

### Vector / Vector Index
A vector is a list of numbers representing something (like the meaning of a text). A vector index is a data structure optimised for quickly finding the most similar vectors to a given query vector. FAISS builds our vector index.
