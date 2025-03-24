from langchain_openai import ChatOpenAI
import streamlit as st

class SummaryGenerator:
    def __init__(self):
        pass
    
    def generate_summary(self, rag_system, api_key, model="gpt-4o-mini", temperature=0.2):
        """
        Generate a hierarchical bullet-point summary from the video transcript
        
        Args:
            rag_system: The RAG system with vector store
            api_key: OpenAI API key
            model: Model to use for summary generation
            temperature: Creativity level (0.0-1.0)
            
        Returns:
            str: Hierarchical bullet-point summary text
        """
        if not rag_system:
            st.error("Please transcribe the video first before creating a summary!")
            return ""
        
        with st.spinner("Generating hierarchical summary..."):
            # Create LLM for summary generation
            summary_llm = ChatOpenAI(
                openai_api_key=api_key,
                model=model,
                temperature=temperature  # Lower temperature for more factual summaries
            )
            
            # Use the RAG system to get relevant context
            try:
                # Get broader context since we're summarizing the whole video
                relevant_docs = rag_system.vector_store.similarity_search(
                    "summarize the main points of this video", k=10
                )
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                prompt = """Based on the video transcript, create a hierarchical bullet-point summary of the content.
                Structure your summary with exactly these levels:
                
                • Main points (use • or * at the start of the line for these top-level points)
                  - Sub-points (use - at the start of the line for these second-level details)
                    * Additional details (use spaces followed by * for third-level points)
                
                For example:
                • First main point
                  - Important detail about the first point
                  - Another important detail
                    * A specific example
                    * Another specific example
                • Second main point
                  - Detail about second point
                
                Be consistent with the exact formatting shown above. Each bullet level must start with the exact character shown (• or *, -, and spaces+*).
                Create 3-5 main points with 2-4 sub-points each, and add third-level details where appropriate.
                Focus on the most important information from the video.
                """
                
                # Use the LLM with context to generate the summary
                messages = [
                    {"role": "system", "content": f"You are given the following context from a video transcript:\n\n{context}\n\nUse this context to create a hierarchical summary according to the instructions."},
                    {"role": "user", "content": prompt}
                ]
                
                response = summary_llm.invoke(messages)
                return response.content
            except Exception as e:
                # Fallback to the regular RAG system if there's an error
                st.warning(f"Using standard summary generation due to error: {str(e)}")
                return rag_system.ask(prompt)
    
    def create_summary_popup_html(self, summary_content):
        """
        Create HTML for the summary popup with properly formatted hierarchical bullets
        
        Args:
            summary_content: Raw summary text with markdown bullet formatting
            
        Returns:
            str: HTML for the popup with properly formatted bullets
        """
        # Instead of relying on markdown conversion, let's manually parse and format the bullet points
        lines = summary_content.strip().split('\n')
        formatted_html = []
        
        in_list = False
        list_level = 0
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Detect if this is a markdown header
            if line.startswith('# '):
                if in_list:
                    # Close any open lists
                    for _ in range(list_level):
                        formatted_html.append('</ul>')
                    in_list = False
                    list_level = 0
                formatted_html.append(f'<h1>{line[2:]}</h1>')
                continue
                
            # Check line for bullet point markers
            if line.startswith('• ') or line.startswith('* '):
                # Top level bullet
                content = line[2:].strip()
                
                if not in_list:
                    # Start a new list
                    formatted_html.append('<ul class="top-level">')
                    in_list = True
                    list_level = 1
                elif list_level > 1:
                    # Close nested lists to get back to top level
                    for _ in range(list_level - 1):
                        formatted_html.append('</ul></li>')
                    list_level = 1
                else:
                    # Close previous list item if needed
                    if formatted_html and not formatted_html[-1].endswith('</ul></li>') and in_list:
                        formatted_html.append('</li>')
                        
                formatted_html.append(f'<li class="top-level-item">{content}')
                
            elif line.startswith('- '):
                # Second level bullet
                content = line[2:].strip()
                
                if not in_list:
                    # Start new lists
                    formatted_html.append('<ul class="top-level"><li class="top-level-item">Second level items')
                    formatted_html.append('<ul class="second-level">')
                    in_list = True
                    list_level = 2
                elif list_level == 1:
                    # Add a nested list
                    formatted_html.append('<ul class="second-level">')
                    list_level = 2
                elif list_level > 2:
                    # Close deeper nested lists to get to second level
                    for _ in range(list_level - 2):
                        formatted_html.append('</ul></li>')
                    list_level = 2
                else:
                    # Close previous list item if needed
                    if formatted_html and not formatted_html[-1].endswith('</ul></li>') and list_level == 2:
                        formatted_html.append('</li>')
                        
                formatted_html.append(f'<li class="second-level-item">{content}')
                
            elif line.startswith('  * ') or line.startswith('    * '):
                # Third level bullet
                content = line.strip()[2:].strip()
                
                if not in_list:
                    # Start new lists (all levels)
                    formatted_html.append('<ul class="top-level"><li class="top-level-item">Top level')
                    formatted_html.append('<ul class="second-level"><li class="second-level-item">Second level')
                    formatted_html.append('<ul class="third-level">')
                    in_list = True
                    list_level = 3
                elif list_level == 2:
                    # Add a nested list
                    formatted_html.append('<ul class="third-level">')
                    list_level = 3
                elif list_level < 3:
                    # We missed a level, adjust
                    formatted_html.append('<li>Missing level</li>')
                    formatted_html.append('<ul class="third-level">')
                    list_level = 3
                else:
                    # Close previous list item if needed
                    if formatted_html and not formatted_html[-1].endswith('</ul></li>') and list_level == 3:
                        formatted_html.append('</li>')
                        
                formatted_html.append(f'<li class="third-level-item">{content}')
            else:
                # Regular paragraph
                if in_list:
                    # Close any open lists
                    for _ in range(list_level):
                        formatted_html.append('</ul>')
                        if list_level > 1:
                            formatted_html.append('</li>')
                    in_list = False
                    list_level = 0
                formatted_html.append(f'<p>{line}</p>')
        
        # Close any open lists
        if in_list:
            # Close final item
            formatted_html.append('</li>')
            # Close any open lists
            for _ in range(list_level):
                if list_level > 1:
                    formatted_html.append('</ul></li>')
                else:
                    formatted_html.append('</ul>')
        
        summary_html = '\n'.join(formatted_html)
        
        html = f"""
        <div id="summary-popup" class="popup-overlay">
            <div class="popup-content">
                <div class="popup-header">
                    <h2>Hierarchical Summary</h2>
                    <button onclick="closeSummaryPopup()" class="close-button">&times;</button>
                </div>
                <div class="popup-body">
                    {summary_html}
                </div>
            </div>
        </div>
        
        <style>
        .popup-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            z-index: 1000;
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        
        .popup-content {{
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            width: 80%;
            max-width: 800px;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }}
        
        .popup-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}
        
        .close-button {{
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            color: #555;
        }}
        
        .close-button:hover {{
            color: #000;
        }}
        
        /* Style for hierarchical bullets */
        .popup-body ul {{
            padding-left: 20px;
            margin-bottom: 5px;
        }}
        
        .popup-body ul.top-level {{
            list-style-type: disc;
        }}
        
        .popup-body ul.second-level {{
            list-style-type: circle;
            margin-top: 5px;
        }}
        
        .popup-body ul.third-level {{
            list-style-type: square;
            margin-top: 3px;
        }}
        
        .popup-body li.top-level-item {{
            margin-bottom: 12px;
            font-weight: bold;
        }}
        
        .popup-body li.second-level-item {{
            margin-bottom: 8px;
            font-weight: normal;
        }}
        
        .popup-body li.third-level-item {{
            margin-bottom: 5px;
            font-weight: normal;
            font-size: 0.95em;
        }}
        
        .popup-body p {{
            margin-bottom: 10px;
        }}
        </style>
        
        <script>
        function closeSummaryPopup() {{
            document.getElementById('summary-popup').style.display = 'none';
            
            // Send message to Streamlit
            window.parent.postMessage({{
                type: "streamlit:setComponentValue",
                value: true
            }}, "*");
        }}
        </script>
        """
        return html