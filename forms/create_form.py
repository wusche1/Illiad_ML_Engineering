import os
import sys

import yaml
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

CREDS_PATH = (
    "/Users/julianschulz/Projects/AI_safety/ML4G-2.0/meta/feedback_forms/creds.json"
)
TOKEN_PATH = os.path.expanduser("~/.config/google/forms_token.json")
SCOPES = ["https://www.googleapis.com/auth/forms.body"]


def get_service():
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        os.makedirs(os.path.dirname(TOKEN_PATH), exist_ok=True)
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())
    return build("forms", "v1", credentials=creds)


def main():
    path = sys.argv[1]
    with open(path) as f:
        spec = yaml.safe_load(f)

    service = get_service()
    form_id = spec.get("form_id")
    if form_id:
        form = service.forms().get(formId=form_id).execute()
        requests = [
            {"deleteItem": {"location": {"index": i}}}
            for i in reversed(range(len(form.get("items", []))))
        ]
    else:
        form = (
            service.forms()
            .create(
                body={"info": {"title": spec["title"], "documentTitle": spec["title"]}}
            )
            .execute()
        )
        form_id = form["formId"]
        with open(path) as f:
            content = f.read()
        with open(path, "w") as f:
            f.write(f"form_id: {form_id}\n" + content)
        requests = []

    if any("correct" in q for q in spec["questions"]):
        requests.append(
            {
                "updateSettings": {
                    "settings": {"quizSettings": {"isQuiz": True}},
                    "updateMask": "quizSettings.isQuiz",
                }
            }
        )
    requests += [
        {
            "updateFormInfo": {
                "info": {"title": spec["title"], "description": spec["description"]},
                "updateMask": "title,description",
            }
        },
    ]

    items = []
    if spec.get("name_field"):
        items.append(
            {
                "title": "Your name",
                "questionItem": {"question": {"required": True, "textQuestion": {}}},
            }
        )
    for q in spec["questions"]:
        if "scale" in q:
            question = {
                "required": True,
                "scaleQuestion": {
                    "low": q["scale"]["low"],
                    "high": q["scale"]["high"],
                    "lowLabel": q["scale"]["low_label"],
                    "highLabel": q["scale"]["high_label"],
                },
            }
        else:
            question = {
                "required": True,
                "grading": {
                    "pointValue": q.get("points", 1),
                    "correctAnswers": {
                        "answers": [{"value": q["choices"][q["correct"]]}]
                    },
                },
                "choiceQuestion": {
                    "type": "RADIO",
                    "options": [{"value": c} for c in q["choices"]],
                },
            }
        items.append({"title": q["title"], "questionItem": {"question": question}})
    for i, item in enumerate(items):
        requests.append({"createItem": {"item": item, "location": {"index": i}}})

    service.forms().batchUpdate(formId=form_id, body={"requests": requests}).execute()

    print(f"Edit URL:    https://docs.google.com/forms/d/{form_id}/edit")
    print(f"Student URL: {form['responderUri']}")


if __name__ == "__main__":
    main()
