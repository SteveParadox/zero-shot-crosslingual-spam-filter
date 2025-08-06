import requests

url = "http://localhost:8000/predict"

examples = [
    # English
    "Congratulations! You've won a free iPhone!",
    "Your account statement is ready.",
    # French
    "Félicitations ! Vous avez gagné un prix incroyable !",
    "Votre relevé bancaire est disponible.",
    # German
    "Herzlichen Glückwunsch! Sie haben einen Preis gewonnen!",
    "Bitte überprüfen Sie Ihre Kontoaktivitäten.",
    # Indian English / Hindi
    "Dear sir, your KYC is pending. Please complete it now.",
    "आपने एक शानदार इनाम जीता है, अभी दावा करें!",
]

for text in examples:
    response = requests.post(url, json={"text": text})
    if response.status_code == 200:
        result = response.json()
        print(f"Text: {text}")
        print(f"→ Language: {result['language']} | Label: {result['label']} | Confidence: {result['confidence']:.4f}\n")
    else:
        print(f"Request failed for: {text}")
