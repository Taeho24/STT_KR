# Pototype website Install
1. After Cloning this repository, move to the directory and open the terminal or command prompt.
2. By using terminal or command prompt, create virtual environment and activate it.
    ```
    Windows :
    1. python -m venv [가상환경이름]
    2. source [가상환경이름]\Scripts\activate
    ```
    ```
    macOS / Linux :
    1. python3 -m venv [가상환경이름]
    2. source [가상환경이름]/bin/activate
    ```
3. Install required libraries by using requirements.txt
    ```
    pip install -r requirements.txt
    ```

4. Migrate and runserver using manage.py
    ```
    cd [directory containing manage.py]
    python manage.py migrate
    python manage.py runserver
    ```
--- 
# Dependencies
- [Django](https://www.djangoproject.com/)