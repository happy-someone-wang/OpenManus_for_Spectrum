import asyncio

from app.agent.manus import Manus
from app.logger import logger
from RAG.rag_agent import EnhancedRAG,RAGConfig

async def main():
    # Create and initialize Manus agent
    agent = await Manus.create()
    config = RAGConfig()
    rag = EnhancedRAG(config)
    try:
        prompt = input("Enter your prompt: ")
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return
        rag_option = input("Using RAG?(Y/N): ")
        rag_option = rag_option.upper()
        if rag_option not in ['Y','N']:
            logger.warning("Wrong optin provided.")
            rag_option = 'N'

        logger.warning("Processing your request...")
        if rag_option =='Y':
            prompt = rag.rag_generate(prompt)
        await agent.run(prompt)
        logger.info("Request processing completed.")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
    finally:
        # Ensure agent resources are cleaned up before exiting
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
