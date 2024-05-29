import json

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

BEGIN_OF_TEXT = "<|begin_of_text|>"
SYSTEM_HEADER = "<|start_header_id|>system<|end_header_id|>"
USER_HEADER = "<|start_header_id|>user<|end_header_id|>"
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>"
EOD = "<|eot_id|>"

Llama3Prompt = f"""{BEGIN_OF_TEXT}{SYSTEM_HEADER}

You are an helpful assistant that provides concise answers to user questions with as little sentences as possible and at maximum 3 sentences. You do not repeat yourself. You avoid bulleted list or emojis.{EOD}{{chat_history}}{USER_HEADER}

{{input}}{EOD}{ASSISTANT_HEADER}"""

'''
Llama3QAPrompt = f"""{BEGIN_OF_TEXT}{SYSTEM_HEADER}

Use the following conversation history and pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. You do not repeat yourself. You avoid bulleted list or emojis.{EOD}{{chat_history}}{USER_HEADER}

Context: {{context}}

{{question}}{EOD}{ASSISTANT_HEADER}"""
'''
'''
Llama3QAPrompt = f"""{BEGIN_OF_TEXT}{SYSTEM_HEADER}

You are an intelligent artificial triage nurse. Use the following conversation history and triage procedure context. Tell me where I should route the patient. If you don't know the answer, just say that you don't know, don't try to make up an answer. You do not repeat yourself. You avoid bulleted list or emojis.{EOD}{{chat_history}}{USER_HEADER}

Context: {{context}}

{{question}}{EOD}{ASSISTANT_HEADER}"""
'''
Llama3QAPrompt = f"""{BEGIN_OF_TEXT}{SYSTEM_HEADER}

        ## 1. Overview
        -You are a top-tier algorithm designed for understanding how to route a patient based on procedure documentation you are given.
        -If needed, ask for clarifying questions to gather more information about the patient to evaluate how to route them.
        
        ## 2. Procedure Documents
        -Documents typically will contain availability of doctors, keywords to look for, general information, and departments to route to.
        
        ## 3. Response
        -Provide a brief, but detailed response, of how to route a patient. 
        -You will split your response into Thought, Action, Observation and Response. 
        -Use the following XML structure and keep everything strictly within these XML tags.  Remember, the <Response> tag contains what's shown to the user. There should be no content outside these XML blocks:
        
        <Thought> Your internal thought process. </Thought>
        <Action> Your actions or analyses. </Action>
        <Observation> User feedback or clarifications. </Observation>
        <Response> Your communication to the user. This is the only visible portion to the user.</Response>
        
        ## 4. Strict Compliance
        Adhere to your procedure documentation. Provide indepth description of why your thought process may be different from the provided operating document with citations. Non-compliance will result in termination.

        ## 5. Provided Procedure Documentation Knowledge: {{context}}

{{question}}{EOD}{ASSISTANT_HEADER}"""


Llama3CondensedQAPrompt = f"""{BEGIN_OF_TEXT}{SYSTEM_HEADER}

Given the following conversation and the question at the end, rephrase the follow up input to be a standalone question, in the same language as the follow up input. You do not repeat yourself. You avoid bulleted list or emojis.{EOD}{{chat_history}}{USER_HEADER}

{{question}}{EOD}{ASSISTANT_HEADER}"""


Llama3PromptTemplate = PromptTemplate.from_template(Llama3Prompt)
Llama3QAPromptTemplate = PromptTemplate.from_template(Llama3QAPrompt)
Llama3CondensedQAPromptTemplate = PromptTemplate.from_template(Llama3CondensedQAPrompt)


class Llama3ConversationBufferMemory(ConversationBufferMemory):
    @property
    def buffer_as_str(self) -> str:
        return self.get_buffer_string()

    def get_buffer_string(self) -> str:
        # See https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
        string_messages = []
        for m in self.chat_memory.messages:
            if isinstance(m, HumanMessage):
                message = f"""{USER_HEADER}

{m.content}{EOD}"""

            elif isinstance(m, AIMessage):
                message = f"""{ASSISTANT_HEADER}

{m.content}{EOD}"""
            else:
                raise ValueError(f"Got unsupported message type: {m}")
            string_messages.append(message)

        return "".join(string_messages)
