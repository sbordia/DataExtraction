## Setup

1. If you don’t have Python installed, [install it from here](https://www.python.org/downloads/)

2. From an admin elevated console, run "choco install tesseract". This is needed to process images

3. Navigate into the project directory

   ```bash
   $ cd Extraction
   ```

4. Create a new virtual environment manually although doing it from VS will also allow debugging

   ```bash
   $ python -m venv venv
   $ . venv/Scripts/activate
   ```

5. Install the requirements

   ```bash
   $ pip install -r requirements.txt
   ```

6. Make a copy of the example environment variables file

   ```bash
   $ cp .env.example .env
   ```

7. Add your [API key](https://beta.openai.com/account/api-keys) to the newly created `.env` file

8. The default mode is to use FAST API. If useFASTAPI is set to False, config.json will be used to load prompt and file

