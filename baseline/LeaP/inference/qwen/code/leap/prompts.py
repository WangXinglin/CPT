GPQA_cot = "Please show your choice in the answer field with only the choice letter, e.g.,\"ANSWER\": \"C\"."
MATH_cot = "Please reason step by step, and put your final answer within \\boxed{}."
LIVECODEBENCH_cot = "Please reason step by step, then output a complete Python solution enclosed in triple backticks."

GPQA_retry = "Please show your choice in the answer field with only the choice letter, e.g.,\"ANSWER\": \"C\". If you think your previous thinking is incorrect, please start thinking completely from scratch."
MATH_retry = "Please reason step by step, and put your final answer within \\boxed{}. If you think your previous thinking is incorrect, please start thinking completely from scratch."

GPQA_temp = """{problem}
A) {A}
B) {B}
C) {C}
D) {D}

"""

GPQA_answer_prompt = " I should show my choice in the answer field with only the choice letter. </think> ANSWER:"
MATH_answer_prompt = "\n\nOh, I think I have found the final answer.\n\n**Final Answer** \\boxed{"
LIVECODEBENCH_answer_prompt = "\n\nOh, I think I have found the final solution. </think>\n\n### Answer:\nPlease provide the complete Python solution in one ```python code block.\n"
MATH_stop_think = "\n\nOh, I think I have finished thinking. </think>"

SYSTEM_MESSAGE_LIVECODEBENCH = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."
FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."


def format_livecodebench_problem(question_content: str, starter_code: str = None) -> str:
    prompt = f"### Question:\n{question_content}\n\n"
    if starter_code:
        prompt += f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt

leap_prefix_MATH = """Please reason step by step, and when you get some intermediate results, please summarize them enlosed with <summarize> </summarize> and you will get the comments from peers. For example:

<summarize> In short, my current key insights about this problem are: Convert numbers to base 10 and set up the equations for the divisibility condition. Then simplify the equation and solve for \\( b \\). After that, find valid solutions, check for constraints, and sum them up for the final answer. And my current progress is: I have computed and confirmed the expressions for 
\\[
17_b = b + 7
\\]
and
\\[
97_b = 9b + 7.
\\]
I then set up the equation
\\[
9b + 7 = k(b + 7)
\\]
and derived the formula
\\[
b = \\frac{7(k - 1)}{9 - k}.
\\] </summarize> <comment> The comments from peers will be presented here. </comment>

After you get the final answer, return the final answer within \\boxed{}."""


leap_prefix_GPQA = """Please reason step by step, and when you get some intermediate results, please summarize them enlosed with <summarize> </summarize> and you will get the comments from peers. 

After you get the final choice, show your choice in the answer field with only the choice letter, e.g.,\"ANSWER\": \"C\"."""


leap_prefix_LIVECODEBENCH = """Please reason step by step, and when you get useful intermediate results, please summarize them enclosed with <summarize> </summarize> and you will get the comments from peers. Keep each summary focused on reusable programming insights: constraints, algorithm choice, invariants, complexity, edge cases, and implementation details. Do not put final code inside the summary.

After you get the final solution, output a complete Python program enclosed in triple backticks. If starter code is provided, complete the required function/class using that starter code. If no starter code is provided, write a full program that reads from stdin and writes to stdout."""


leap_subfix_MATH = """Okay, so I have this complex mathematical problem. And the user instruct that I should summarize what I've concluded with tags when I get some intermediate results. For example:

<summarize> In short, my current key insights about this problem are: Convert numbers to base 10 and set up the equations for the divisibility condition. Then simplify the equation and solve for \\( b \\). After that, find valid solutions, check for constraints, and sum them up for the final answer. And my current progress is: I have computed and confirmed the expressions for 
\\(
17_b = b + 7
\\)
and
\\(
97_b = 9b + 7.
\\)
I then set up the equation
\\(
9b + 7 = k(b + 7)
\\)
and derived the formula
\\(
b = \\frac{7(k - 1)}{9 - k}.
\\) </summarize>

Now, let's get back to the original problem."""


leap_subfix_GPQA = """Okay, so I have this complex problem. And the user instruct that I should summarize what I've concluded with tags when I get some intermediate results.

Now, let's get back to the original problem."""


leap_subfix_LIVECODEBENCH = """Okay, so I have this programming problem. And the user instruct that I should summarize what I've concluded with tags when I get useful intermediate results.

Now, let's get back to the original programming problem."""


leap_triggers = [
    "Alright, let's take a step back and summarize what we've figured out so far briefly.",
    "Wait, let me quickly recap what I've concluded so far.",
    "Alright, let me shortly review the conclusions I've drawn so I can move forward more efficiently.",
    "Hmm, a quick summary of what I've figured out might help streamline the next part of my reasoning.",
    "Hold on, I should summarize the key points briefly to ensure I'm on the right track.",
    "Okay, before continuing, let me put together a brief summary of the insights I've gathered so far.",
    "Okay, time to consolidate everything I've found into a concise summary."
]


summarize_triggers = [
    " <summarize> In short, my current conclusions are that",
    " <summarize> To summarize, based on my previous reasoning, I have currently found that",
    " <summarize> In conclusion, the current key takeaways and results are",
    " <summarize> In short, I've currently concluded that",
    " <summarize> To summarize, my recent findings are",
    " <summarize> In conclusion, the current insights and results I've gathered are",
]

comment_subfix = "Hmm, it seems that my peers have given me some comments, so let me check if anyone's conclusions are different from mine before I continue my own reasoning."

moa_template = """Problem: {problem}

You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:
"""

def get_leap():
    import random
    return random.choice(leap_triggers) + random.choice(summarize_triggers)
