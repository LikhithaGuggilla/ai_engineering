# AI-powered-shoping-assistant (Amazon)

Navigating a vast e-commerce platform like Amazon presents significant challenges for customers. They often face information overload when trying to choose the best product, spending considerable time sifting through thousands of product descriptions, specifications, and often contradictory customer reviews. This process is inefficient and can lead to choice paralysis or poor purchasing decisions. There is a clear need for an intelligent, interactive solution that can instantly synthesize this vast amount of information, provide personalized guidance, and streamline the shopping process from discovery to checkout.

This project aims to build an AI-powered chatbot assistant that directly addresses these gaps by providing immediate, context-aware answers, summarizing reviews, comparing products, and executing tasks on the user's behalf.

## High-Level Architecture
![architecture](https://github.com/LikhithaGuggilla/ai_engineering/blob/main/system-architecture.png)
The system will be composed of several key, interacting components:
1.	Frontend (Streamlit UI): A simple, clean chat interface where the user interacts with the assistant.
2.	Backend (FastAPI)/ Orchestrator: The central logic hub of the application. It will:

        ○	Receive user queries from the frontend.
  	    ○	Recognize user intent (is it a question or a task?) and generate multi-step execution plan.
  	    ○	Invoke the Retrieval module to fetch relevant context from the Vector DB.
  	    ○	Invoke the LLM to generate a response.
  	    ○	Execute tools/APIs for task-oriented requests using MCP/A2A communication protocols.

4.	Vector Database (Qdrant): An indexed, searchable knowledge base containing the vector embeddings of all product information and reviews.
5.	LLM Service (Gemini, OpenAI, Groq): An API endpoint for a pre-trained Large Language Model that handles the natural language generation.
6.	Tools / API Layer: A set of functions that connect to external services. Initially, these will be mock APIs like addToCart(product_id) and placeOrder(cart_details). The Human-in-the-Loop (HITL) mechanism will be triggered here for sensitive actions.

## Performance Metrics & Evaluation
To measure the assistant's effectiveness, below metrics are tracked:

- Performance Metrics
  
      ●	User Satisfaction (CSAT): A simple "thumbs up/thumbs down" feedback mechanism after each response to gauge user sentiment.
      
      ●	Latency: The time from when a user sends a query to when they receive a response. We will aim for a response time of under 3 seconds.

- Evaluation

      ●	Response Quality:
      
          ○	Faithfulness: How factually accurate is the generated response based on the retrieved context? (Scale 1-5)
          
          ○	Answer Relevancy: How relevant is the response to the user's question? (Scale 1-5)
      
      ●	Task Success Rate: The percentage of times the agent successfully completes a requested task (e.g., correctly adding the specified item to the cart).

