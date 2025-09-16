import os
import logging
import requests
import json
import datetime
from typing import Dict, Any, Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from openai import OpenAI

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configure file logging for requests and results
LOG_DIR = os.environ.get("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Create a file handler for the request logger
request_logger = logging.getLogger("request_logger")
request_logger.setLevel(logging.INFO)

# Create log filename with current date
log_filename = os.path.join(LOG_DIR, f"telegram_requests_{datetime.datetime.now().strftime('%Y-%m-%d')}.log")
file_handler = logging.FileHandler(log_filename)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
request_logger.addHandler(file_handler)

# Environment variables
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
SEARCH_URL = os.environ.get("SEARCH_URL", "http://localhost:8000")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LLM_URL = os.environ.get("LLM_URL", "http://localhost:1234/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")

# Global OpenAI client
llm_client = None

# Initialize OpenAI client
def initialize_llm_client():
    global llm_client

    if OPENAI_API_KEY:
        logger.info("Initializing OpenAI client with API key")
        llm_client = OpenAI(api_key=OPENAI_API_KEY)
    elif LLM_URL:
        logger.info(f"Initializing OpenAI client with custom base URL: {LLM_URL}")
        llm_client = OpenAI(base_url=LLM_URL, api_key="dummy_key")
    else:
        logger.warning("No LLM configuration provided (OPENAI_API_KEY or LLM_URL)")
        llm_client = None

    return llm_client is not None

# Search function
async def search_knowledge_base(query: str, top_k: int = 5, use_hybrid: bool = True) -> list:
    """Search the knowledge base using the indexer API"""
    try:
        search_params = {
            "query": query,
            "top_k": top_k,
            "use_hybrid": use_hybrid,
        }

        response = requests.post(
            f"{SEARCH_URL}/search",
            json=search_params,
            timeout=30
        )
        response.raise_for_status()

        return response.json()["results"]
    except Exception as e:
        logger.error(f"Error searching knowledge base: {str(e)}")
        return []

# Generate response using LLM
async def generate_llm_response(query: str, context_data: list) -> str:
    """Generate response using LLM"""
    try:
        if llm_client is None:
            return "Sorry, the LLM service is not available at the moment."

        # Prepare context from retrieved QA pairs
        context = ""
        for i, item in enumerate(context_data):
            context += f"Context {i+1}:\nQuestion: {item['question']}\nAnswer: {item['answer']}\n\n"

        # Create prompt
        prompt = f"""Ты эксперт-криминалист. Используйте приведённые ниже контекстные материалы (Context 1, Context 2 и т. д.) для ответа на вопрос пользователя.
Если на основании контекста невозможно сформулировать ответ, скажите «У меня недостаточно информации, чтобы ответить на этот вопрос».

{context}

User Question: {query}

Answer:"""

        # Log which LLM model we're using
        logger.info(f"Using LLM model: {LLM_MODEL}")

        # Make the API call using the global client
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error generating LLM response: {str(e)}")
        return f"Извините, при формировании ответа произошла ошибка: {str(e)}"

# Command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        f"Привет  {user.mention_html()}! Я твой помощник RAG. Задай мне любой вопрос, и я постараюсь найти ответ в своей базе знаний. Введи /help для подробностей."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    help_text = """
*Помощь с ботом RAG по криминалистике*

Этот бот использует технологию расширенной генерации данных (RAG), чтобы отвечать на ваши вопросы на основе базы знаний.

*Commands:*
/start - Start the bot
/help - Показать это справочное сообщение
/settings - Настройте параметры поиска

*Как использовать:*
Просто введите свой вопрос, и я найду соответствующую информацию и сгенерирую ответ.

*Настройки поиска:*
Ты можешь настроить способ поиска информации с помощью команды /settings.
- Гибридный поиск: объединяет векторный поиск и поиск по ключевым словам
    """
    await update.message.reply_markdown(help_text)

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /settings command."""
    keyboard = [
        [
            InlineKeyboardButton("Гибридный поиск: ON", callback_data="hybrid_on"),
            InlineKeyboardButton("Гибридный поиск: OFF", callback_data="hybrid_off"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Настроить поиск:", reply_markup=reply_markup)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button callbacks."""
    query = update.callback_query
    await query.answer()

    # Get user data or initialize if not exists
    if not context.user_data.get("search_settings"):
        context.user_data["search_settings"] = {
            "use_hybrid": True,
        }

    # Handle different callbacks
    if query.data == "hybrid_on":
        context.user_data["search_settings"]["use_hybrid"] = True
        await query.edit_message_text("Гибридный поиск переключен ON")

    elif query.data == "hybrid_off":
        context.user_data["search_settings"]["use_hybrid"] = False
        await query.edit_message_text("Гибридный поиск переключен OFF")

    elif query.data == "settings_done":
        settings = context.user_data["search_settings"]
        settings_text = f"""
*Текущие настройки:*
- Гибридный поиск: {'ON' if settings['use_hybrid'] else 'OFF'}
        """
        await query.edit_message_text(settings_text, parse_mode="Markdown")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle user messages."""
    # Get user's question
    question = update.message.text
    user_id = update.effective_user.id
    username = update.effective_user.username or "unknown"

    # Log the user request
    request_logger.info(f"USER_REQUEST | User: {username} (ID: {user_id}) | Question: {question}")

    # Get search settings from user data or use defaults
    search_settings = context.user_data.get("search_settings", {
        "use_hybrid": True,
    })

    # Send typing action
    await update.message.chat.send_action("typing")

    # First, let the user know we're processing their question
    processing_message = await update.message.reply_text("🔍 Ищу информацию...")

    try:
        # Search for relevant information
        search_results = await search_knowledge_base(
            query=question,
            top_k=3,
            use_hybrid=search_settings["use_hybrid"],
        )

        # Log search results
        search_results_log = json.dumps(search_results, ensure_ascii=False) if search_results else "No results found"
        request_logger.info(f"SEARCH_RESULTS | User: {username} (ID: {user_id}) | Query: {question} | Results: {search_results_log}")

        if not search_results:
            await processing_message.edit_text("Мне не удалось найти никакой информации, которая могла бы ответить на ваш вопрос.")
            return

        # Update message to show we're generating a response
        await processing_message.edit_text("🤔 Создание ответа...")

        # Generate response using LLM
        answer = await generate_llm_response(question, search_results)

        # Log the generated response
        request_logger.info(f"GENERATED_RESPONSE | User: {username} (ID: {user_id}) | Question: {question} | Response: {answer[:500]}{'...' if len(answer) > 500 else ''}")

        # Prepare sources text
        sources_text = "\n\n*Источники:*\n"
        for i, source in enumerate(search_results):
            # Truncate long answers
            short_answer = source["answer"][:100] + "..." if len(source["answer"]) > 100 else source["answer"]
            sources_text += f"{i+1}. Q: {source['question']}\nA: {short_answer}\n"

        # Send the final answer with sources
        response_text = f"{answer}{sources_text}"

        # If response is too long, split it
        if len(response_text) > 4000:
            await processing_message.edit_text(answer[:4000])
            await update.message.reply_text(sources_text[:4000], parse_mode="Markdown")
        else:
            await processing_message.edit_text(response_text, parse_mode="Markdown")

    except Exception as e:
        error_message = f"Error processing message: {str(e)}"
        logger.error(error_message)
        request_logger.error(f"ERROR | User: {username} (ID: {user_id}) | Question: {question} | Error: {error_message}")
        await processing_message.edit_text(f"Извините, произошла ошибка: {str(e)}")

def main() -> None:
    """Start the bot."""
    # Check if token is provided
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN environment variable is not set!")
        return

    # Initialize LLM client
    if not initialize_llm_client():
        logger.warning("Failed to initialize LLM client. Bot will start but may not generate responses.")

    # Create the Application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("settings", settings_command))
    application.add_handler(CallbackQueryHandler(button_callback))

    # Handle messages
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run the bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
