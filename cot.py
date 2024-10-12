import streamlit as st
import groq
import os
import json
import time
import instructor
from anthropic import Anthropic
from pydantic import BaseModel
from openai import OpenAI

# Pydantic model for Anthropic responses
class StepResponse(BaseModel):
    title: str
    content: str
    next_action: str
    confidence: float

# Initialize clients (will be set later when API keys are provided)
groq_client = None
anthropic_client = None
openai_client = None

def make_groq_call(messages, max_tokens, is_final_answer=False):
    global groq_client
    if groq_client is None:
        return {"title": "Error", "content": "Groq client is not initialized. Please check your API key.", "next_action": "final_answer"}
    
    for attempt in range(3):
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt == 2:
                if is_final_answer:
                    return {"title": "Error", "content": f"Failed to generate final answer after 3 attempts. Error: {str(e)}"}
                else:
                    return {"title": "Error", "content": f"Failed to generate step after 3 attempts. Error: {str(e)}", "next_action": "final_answer"}
            time.sleep(1)  # Wait for 1 second before retrying

def make_anthropic_call(system_prompt, messages, max_tokens, is_final_answer=False):
    global anthropic_client
    if anthropic_client is None:
        return StepResponse(
            title="Error",
            content="Anthropic client is not initialized. Please check your API key.",
            next_action="final_answer",
            confidence=0.5
        )
    
    for attempt in range(3):
        try:
            response = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=max_tokens,
                temperature=0.2,
                system=system_prompt,
                messages=messages,
                response_model=StepResponse
            )
            return response
        except Exception as e:
            if attempt == 2:
                return StepResponse(
                    title="Error",
                    content=f"Failed to generate {'final answer' if is_final_answer else 'step'} after 3 attempts. Error: {str(e)}",
                    next_action="final_answer",
                    confidence=0.5
                )
            time.sleep(1)  # Wait for 1 second before retrying

def make_openai_call(messages, max_tokens, is_final_answer=False):
    global openai_client
    if openai_client is None:
        return {"title": "Error", "content": "OpenAI client is not initialized. Please check your API key.", "next_action": "final_answer"}
    
    for attempt in range(3):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt == 2:
                if is_final_answer:
                    return {"title": "Error", "content": f"Failed to generate final answer after 3 attempts. Error: {str(e)}"}
                else:
                    return {"title": "Error", "content": f"Failed to generate step after 3 attempts. Error: {str(e)}", "next_action": "final_answer"}
            time.sleep(1)  # Wait for 1 second before retrying

def generate_response(prompt, api_choice):
    system_prompt = """You are an AI assistant that explains your reasoning step by step, incorporating dynamic Chain of Thought (CoT), reflection, and verbal reinforcement learning. Follow these instructions:

1. Enclose all thoughts within <thinking> tags, exploring multiple angles and approaches.
2. Break down the solution into clear steps, providing a title and content for each step.
3. After each step, decide if you need another step or if you're ready to give the final answer.
4. Continuously adjust your reasoning based on intermediate results and reflections, adapting your strategy as you progress.
5. Regularly evaluate your progress, being critical and honest about your reasoning process.
6. Assign a quality score between 0.0 and 1.0 to guide your approach:
   - 0.8+: Continue current approach
   - 0.5-0.7: Consider minor adjustments
   - Below 0.5: Seriously consider backtracking and trying a different approach
7. If unsure or if your score is low, backtrack and try a different approach, explaining your decision.
8. For mathematical problems, show all work explicitly using LaTeX for formal notation and provide detailed proofs.
9. Explore multiple solutions individually if possible, comparing approaches in your reflections.
10. Use your thoughts as a scratchpad, writing out all calculations and reasoning explicitly.
11. Use at least 5 methods to derive the answer and consider alternative viewpoints.
12. Be aware of your limitations as an AI and what you can and cannot do.

After every 3 steps, perform a detailed self-reflection on your reasoning so far, considering potential biases and alternative viewpoints."""

    messages = [
        {"role": "user", "content": prompt}
    ]
    
    steps = []
    step_count = 1
    total_thinking_time = 0
    
    while True:
        start_time = time.time()
        if api_choice == "Groq (LLAMA 3.1 8b)":
            step_data = make_groq_call(messages, 750)
        elif api_choice == "Anthropic (Claude)":
            step_data = make_anthropic_call(system_prompt, messages, 750)
            step_data = step_data.model_dump()
        else:  # OpenAI
            step_data = make_openai_call(messages, 750)
        
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time
        
        confidence = step_data.get('confidence', 0.5)
        
        steps.append((f"Step {step_count}: {step_data.get('title', 'Untitled Step')}", 
                      step_data.get('content', 'No content provided'), 
                      thinking_time, 
                      confidence))
        
        messages.append({"role": "assistant", "content": json.dumps(step_data)})
        
        next_action = step_data.get('next_action', 'continue')
        
        if next_action == 'final_answer' and step_count < 15:
            messages.append({"role": "user", "content": "Please continue your analysis with at least 5 more steps before providing the final answer."})
        elif next_action == 'final_answer':
            break
        elif next_action == 'reflect' or step_count % 3 == 0:
            messages.append({"role": "user", "content": "Please perform a detailed self-reflection on your reasoning so far, considering potential biases and alternative viewpoints."})
        else:
            messages.append({"role": "user", "content": "Please continue with the next step in your analysis."})
        
        step_count += 1

        yield steps, None

    messages.append({"role": "user", "content": "Please provide a comprehensive final answer based on your reasoning above, summarizing key points and addressing any uncertainties."})
    
    start_time = time.time()
    if api_choice == "Groq (LLAMA 3.1 8b)":
        final_data = make_groq_call(messages, 750, is_final_answer=True)
    elif api_choice == "Anthropic (Claude)":
        final_data = make_anthropic_call(system_prompt, messages, 750, is_final_answer=True)
        final_data = final_data.model_dump()
    else:  # OpenAI
        final_data = make_openai_call(messages, 750, is_final_answer=True)
    
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time
    
    final_confidence = final_data.get('confidence', 1.0)
    
    steps.append(("Final Answer", final_data.get('content', 'No final answer provided'), thinking_time, final_confidence))

    yield steps, total_thinking_time

def main():
    st.set_page_config(page_title="AI Reasoning Chain", page_icon="🧠", layout="wide")
    
    st.title("AI Reasoning Chain: Extended self-reflection and analysis")
    
    st.markdown("""
    This application demonstrates an advanced AI reasoning chain with extended self-reflection to improve output accuracy. 
    It leverages different AI models to provide detailed, step-by-step analysis of complex queries.

    ### How it works:
    1. The AI breaks down the problem into multiple steps.
    2. For each step, it provides a title, content, and confidence score.
    3. The AI performs self-reflection every few steps to evaluate its progress.
    4. Finally, it provides a comprehensive answer based on its analysis.

    ### Features:
    - Supports multiple AI providers: Groq (LLAMA 3.1 8b), Anthropic (Claude), and OpenAI (GPT-4)
    - Dynamic Chain of Thought (CoT) reasoning
    - Extended thinking time for more thorough analysis
    - Self-reflection and bias consideration

    ### Enhancing Smaller Models:
    This technique significantly improves the logical reasoning capabilities of smaller, faster, and more cost-effective models:

    1. **LLAMA 3.1 8b**: Despite its compact size, this model can produce surprisingly sophisticated reasoning when guided by our structured approach.
    2. **GPT-4 Turbo**: This model balances speed and capability, and our technique helps it achieve more consistent and reliable results.

    By breaking down complex problems and encouraging step-by-step analysis, even these smaller models can tackle challenging logical reasoning tasks more effectively.

     ### Claude Sonnet 3.5 vs OpenAI's o1-preview Strawberry:
    Using this technique, Claude's Sonnet 3.5 model has shown superior performance in logical reasoning tasks compared to OpenAI's Strawberry o1-preview model. Here's why:

    1. **Structured Reasoning**: Claude excels at following structured prompts, allowing for more consistent step-by-step analysis.
    2. **Self-Reflection**: The built-in self-reflection mechanism works particularly well with Claude, enabling it to catch and correct errors mid-reasoning.
    3. **Explicit Uncertainty Handling**: Claude is adept at expressing and quantifying uncertainty, leading to more nuanced and accurate conclusions.
    4. **Mathematical Reasoning**: For tasks involving mathematical or formal logic, Claude often provides more rigorous and detailed proofs.
    5. **Bias Awareness**: Claude's self-reflection steps are particularly effective at identifying and mitigating potential biases in its reasoning.

    While both models are powerful, this technique leverages Claude's strengths in structured thinking and self-analysis, often resulting in more robust and reliable logical reasoning for complex queries.

    ### Credits:
    This advanced reasoning technique was developed and is maintained by [MultipleWords](https://multiplewords.com). 
    We are grateful for their innovative work in improving AI reasoning capabilities across various model sizes and architectures.
    """)

    st.markdown("### Steps to use:")
    st.markdown("""
    1. Choose your preferred AI provider using the radio buttons below.
    2. Enter the API key for the selected provider in the text input field.
    3. Type your query in the text box provided.
    4. Click outside the text box or press Enter to start the analysis.
    5. Wait for the AI to generate a response (this may take a few minutes).
    6. Explore the step-by-step reasoning and final answer provided by the AI.
    7. Compare results between different models to see the differences in reasoning approaches.
    8. Notice how even smaller models can produce sophisticated reasoning with this technique.
    """)
    
    # API choice
    api_choice = st.radio("Choose AI Provider:", ("Groq (LLAMA 3.1 8b)", "Anthropic (Claude)", "OpenAI (GPT-4 Turbo)"))
    
    # API key input based on selection
    global groq_client, anthropic_client, openai_client
    api_key = None
    if api_choice == "Groq (LLAMA 3.1 8b)":
        api_key = st.text_input("Enter your Groq API key:", type="password", help="You can obtain a Groq API key from https://console.groq.com/")
        if api_key:
            groq_client = groq.Groq(api_key=api_key)
    elif api_choice == "Anthropic (Claude)":
        api_key = st.text_input("Enter your Anthropic API key:", type="password", help="You can obtain an Anthropic API key from https://console.anthropic.com/")
        if api_key:
            anthropic_client = instructor.from_anthropic(Anthropic(api_key=api_key), mode=instructor.mode.Mode.ANTHROPIC_JSON)
    else:  # OpenAI
        api_key = st.text_input("Enter your OpenAI API key:", type="password", help="You can obtain an OpenAI API key from https://platform.openai.com/api-keys")
        if api_key:
            openai_client = OpenAI(api_key=api_key)
    
    # Text input for user query
    user_query = st.text_input("Enter your query:", placeholder="e.g., What are the potential long-term effects of climate change on global agriculture?")
    
    if user_query and api_key:
        st.write("Generating response... This may take a while due to extended thinking time.")
        
        # Create empty elements to hold the generated text and total time
        response_container = st.empty()
        time_container = st.empty()
        
        # Generate and display the response
        for steps, total_thinking_time in generate_response(user_query, api_choice):
            with response_container.container():
                for i, (title, content, thinking_time, confidence) in enumerate(steps):
                    if title.startswith("Final Answer"):
                        st.markdown(f"### {title}")
                        st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)
                        st.markdown(f"**Confidence:** {confidence:.2f}")
                    else:
                        with st.expander(title, expanded=True):
                            st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)
                            st.markdown(f"**Confidence:** {confidence:.2f}")
                            st.markdown(f"**Thinking time:** {thinking_time:.2f} seconds")
            
            # Only show total time when it's available at the end
            if total_thinking_time is not None:
                time_container.markdown(f"**Total thinking time: {total_thinking_time:.2f} seconds**")
    elif user_query:
        st.error(f"Please enter a valid API key for {api_choice}.")
    else:
        st.info("Enter your query above and provide an API key to get started.")

    st.markdown("---")
    st.markdown("""Powered by the advanced reasoning technique developed by [MultipleWords](https://multiplewords.com)
                
                  For support or inquiries, contact: taher@multiplewords.com
                """)

if __name__ == "__main__":
    main()