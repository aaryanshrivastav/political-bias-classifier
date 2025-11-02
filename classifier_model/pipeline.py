from transformers import pipeline

export_dir = r"C:\Projects\political-bias-classifier-1\classifier_model\models\final_models"

classifier = pipeline("text-classification", model=export_dir)
print(classifier("The government must prioritize social welfare programs and increase taxes on the wealthy to ensure equality."))
print(classifier("The government announced a new budget plan for the upcoming fiscal year."))
print(classifier("The government announced a new budget plan.Cutting taxes and reducing government regulation will strengthen the economy and create jobs."))
