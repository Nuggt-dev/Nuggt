text = "```hello\nworld```"
print(text.strip("```"))
print(text.strip("```").strip("hello\n"))