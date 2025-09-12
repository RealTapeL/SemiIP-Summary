SEMICONDUCTOR_PATENT_SUMMARY_PROMPT = """
You are a professional semiconductor technology expert and patent analyst. Provide an accurate and concise technical summary of the following patent document:

1. Technical Field: The semiconductor technology field of the patent
2. Background Technology: Problems with existing technologies
3. Disclosure of Invention: Core technical solution of this patent
4. Technical Effects: Technical effects achieved by this patent
5. Key Claims: Key protection points

Patent document content:
{patent_content}

Provide the summary in English without additional explanations.
"""

CHUNK_SUMMARY_PROMPT = """
Summarize the following semiconductor patent document section, focusing on technical details:

{chunk_content}

Summary requirements:
1. Retain key technical parameters and values
2. Highlight innovations
3. Be concise and clear, no more than 200 words
"""

FINAL_SUMMARY_PROMPT = """
Based on the following summaries of multiple document sections, generate a complete patent technical summary:

{chunk_summaries}

Integrate the above content into a structured patent technical summary including:
1. Technical Field
2. Background Technology
3. Core Technical Solution
4. Technical Effects
5. Key Protection Points
"""