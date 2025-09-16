import re
import pandas as pd
import os

def read_markdown_file(file_path):
    """Чтение содержимого markdown-файла"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    return content

def markdown_to_excel(markdown_text=None, markdown_file=None, output_file='output.xlsx'):
    """Конвертация markdown в Excel с колонками question-answer"""
    
    # Получаем текст либо из параметра, либо из файла
    if markdown_file:
        markdown_text = read_markdown_file(markdown_file)
    elif not markdown_text:
        raise ValueError("Необходимо указать либо текст markdown, либо путь к файлу")
    
    # Разделяем текст на блоки вопрос-ответ
    pattern = r'# (.*?)\n([\s\S]*?)(?=# |$)'
    matches = re.findall(pattern, markdown_text)
    
    # Создаем DataFrame
    data = {
        'question': [],
        'answer': []
    }
    
    for match in matches:
        question = match[0].strip()
        answer = match[1].strip()
        data['question'].append(question)
        data['answer'].append(answer)
    
    # Создаем DataFrame и сохраняем в Excel
    df = pd.DataFrame(data)
    df.to_excel(output_file, index=False)
    
    return f"Файл {output_file} успешно создан с {len(data['question'])} вопросами и ответами."

def main():
    try:
        result = markdown_to_excel(markdown_file='База для дообучения модели.md', output_file='qa_database.xlsx')
        print(result)
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
