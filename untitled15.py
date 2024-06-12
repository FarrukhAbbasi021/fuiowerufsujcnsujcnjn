# -*- coding: utf-8 -*-
"""Untitled15.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1M7E95uDXC5PJ81M5ZLrB1bSJpv2FXgOa
"""

{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5dd8becb",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pinecone\n",
    "import sys\n",
    "import time\n",
    "import traceback\n",
    "import openai\n",
    "import streamlit as st\n",
    "from streamlit_chat import message\n",
    "import streamlit.components.v1 as components\n",
    "from langchain.vectorstores import Pinecone\n",
    "from langchain.prompts import PromptTemplate\n",
    "from langchain.embeddings.openai import OpenAIEmbeddings\n",
    "from langchain.chat_models import ChatOpenAI\n",
    "from langchain.chains.summarize import load_summarize_chain\n",
    "from langchain.memory import ConversationSummaryBufferMemory\n",
    "from langchain.chains import ConversationalRetrievalChain\n",
    "from langchain.chains.llm import LLMChain\n",
    "from langchain.chains.combine_documents.stuff import StuffDocumentsChain\n",
    "from langchain.prompts.chat import (\n",
    "    ChatPromptTemplate,\n",
    "    SystemMessagePromptTemplate,\n",
    "    AIMessagePromptTemplate,\n",
    "    HumanMessagePromptTemplate,\n",
    ")\n",
    "\n",
    "try:\n",
    "    import environment_variables\n",
    "except ImportError:\n",
    "    pass\n",
    "\n",
    "element_id = 0\n",
    "\n",
    "\n",
    "def sanitize_answer(raw_answer):\n",
    "    answer = ''\n",
    "    chunks = raw_answer.split('\\n')\n",
    "    sub_string_list = [\n",
    "        '- SOURCE:', '- Source:', '- source:', '- SOURCES:', '- Sources:', '- sources:',\n",
    "        '(SOURCE:', '(Source:', '(source:', '(SOURCES:', '(Sources:', '(sources:',\n",
    "        'SOURCE:', 'Source:', 'source:', 'SOURCES:', 'Sources:', 'sources:'\n",
    "    ]\n",
    "\n",
    "    try:\n",
    "        for chunk in chunks:\n",
    "            temp_string = str(chunk).strip()\n",
    "            temp_string_lowercase = temp_string.lower()\n",
    "            answer_text = ''\n",
    "\n",
    "            if 'source' in temp_string_lowercase:\n",
    "                for sub_string in sub_string_list:\n",
    "                    if sub_string in temp_string:\n",
    "                        temp_string = temp_string[:temp_string.index(sub_string)]\n",
    "\n",
    "            answer_text = temp_string.strip()\n",
    "            if answer_text:\n",
    "                answer += '\\n\\n' + answer_text\n",
    "\n",
    "        answer = answer.strip()\n",
    "    except Exception as e:\n",
    "        handle_error(e)\n",
    "\n",
    "    return answer\n",
    "\n",
    "\n",
    "def get_source_list(sources):\n",
    "    chunks = []\n",
    "    source_list = []\n",
    "    allowed_file_extensions = ['.pdf']\n",
    "\n",
    "    try:\n",
    "        if '\\n' in sources:\n",
    "            chunks = sources.split('\\n')\n",
    "        elif ',' in sources:\n",
    "            chunks = sources.split(',')\n",
    "        else:\n",
    "            source = sources.strip()\n",
    "            chunks.append(source)\n",
    "\n",
    "        for chunk in chunks:\n",
    "            chunk = chunk.strip()\n",
    "            if len(chunk) >= 5:\n",
    "                if chunk.startswith('- '):\n",
    "                    chunk = chunk[2:]\n",
    "                extension = os.path.splitext(chunk)[1].lower()\n",
    "\n",
    "                if extension in allowed_file_extensions:\n",
    "                    source_list.append(chunk)\n",
    "    except Exception as e:\n",
    "        handle_error(e)\n",
    "\n",
    "    return source_list\n",
    "\n",
    "\n",
    "def accordion(query, sources, answer_type):\n",
    "    html = ''\n",
    "    global element_id\n",
    "    sources = sources.strip()\n",
    "    source_id = 0\n",
    "    accordion_height = 0\n",
    "\n",
    "    try:\n",
    "        if len(sources) < 5:\n",
    "            return html, accordion_height\n",
    "        else:\n",
    "            source_list = get_source_list(sources)\n",
    "\n",
    "            if answer_type == \"Chat on projects with sources and summaries\":\n",
    "                model_name = 'text-embedding-ada-002'\n",
    "\n",
    "                embed = OpenAIEmbeddings(\n",
    "                    model=model_name,\n",
    "                    openai_api_key=OPENAI_API_KEY\n",
    "                )\n",
    "\n",
    "                text_field = \"text\"\n",
    "\n",
    "                vectorstore = Pinecone(\n",
    "                    child_index, embed.embed_query, text_field\n",
    "                )\n",
    "\n",
    "                llm = ChatOpenAI(\n",
    "                    openai_api_key=OPENAI_API_KEY,\n",
    "                    model_name=model,\n",
    "                    temperature=0.0,\n",
    "                    verbose=False,\n",
    "                )\n",
    "\n",
    "        html += '<link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css\" rel=\"stylesheet\">'\n",
    "        html += '<script src=\"https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js\"></script>'\n",
    "        html += '<div class=\"m-4\">'\n",
    "        html += '<div class=\"accordion\" id=\"myAccordion\">'\n",
    "\n",
    "        for source in source_list:\n",
    "            source = source.strip()\n",
    "            source_id += 1\n",
    "            element_id += 1\n",
    "\n",
    "            if answer_type == \"Chat on projects with sources\":\n",
    "                summary = \"\"\n",
    "            elif answer_type == \"Chat on projects with sources and summaries\":\n",
    "                summary_text = \"\"\n",
    "                summary = \"\"\n",
    "\n",
    "                retriever = vectorstore.as_retriever(\n",
    "                    search_kwargs={\"k\": 9, 'filter': {'source': source}})\n",
    "                docs = retriever.get_relevant_documents(query)\n",
    "\n",
    "                prompt_template_summ = \"\"\"Write a concise summary of the following, be sure to include the aims, lessons learnt and conclusions: \"{text_sum}\"\n",
    "                ALWAYS give your answer in point form like in the example.\n",
    "                \n",
    "                Example:\n",
    "                Aim: This project aimed to determine if videoconsult was superior to face to face consults \\n\n",
    "\n",
    "                \n",
    "                Lessons Learnt: None found \\n\n",
    "\n",
    "                \n",
    "                Conclusions: Patients preferred videoconsults to physical consults.\n",
    "                \n",
    "                \n",
    "                #CONCISE SUMMARY:\"\"\"\n",
    "                prompt_summ = PromptTemplate.from_template(\n",
    "                    prompt_template_summ)\n",
    "                llm_chain_summ = LLMChain(llm=llm, prompt=prompt_summ)\n",
    "                stuff_chain = StuffDocumentsChain(\n",
    "                    llm_chain=llm_chain_summ, document_variable_name=\"text_sum\")\n",
    "                summary_text = stuff_chain.run(docs)\n",
    "\n",
    "                summary_text = summary_text.strip().replace(\"\\n\", \"<br>\")\n",
    "                if summary_text:\n",
    "                    summary = f\" <br><strong>Summary:</strong> <br> {summary_text}\"\n",
    "            else:\n",
    "                summary = \"\"\n",
    "\n",
    "            html += '<div class=\"accordion-item\">'\n",
    "            html += f'<h2 class=\"accordion-header\" id=\"heading-{source_id}\">'\n",
    "            html += f'<button type=\"button\" class=\"accordion-button collapsed\" data-bs-toggle=\"collapse\" data-bs-target=\"#collapse-{source_id}\">Source {source_id}</button>'\n",
    "            html += '</h2>'\n",
    "            html += f'<div id=\"collapse-{source_id}\" class=\"accordion-collapse collapse\" data-bs-parent=\"#myAccordion\">'\n",
    "            html += '<div class=\"card-body\">'\n",
    "            html += f'<p><strong><a href=\"https://child.chi.sg/files/pdffiles/{source}\" target=\"_blank\"><span style=\"overflow-wrap: break-word;\">{source}</span></a></strong>{summary}</p>'\n",
    "            html += '</div>'\n",
    "            html += '</div>'\n",
    "            html += '</div>'\n",
    "\n",
    "        html += '</div>'\n",
    "        html += '</div>'\n",
    "\n",
    "        if answer_type == \"Chat on projects with sources\":\n",
    "            accordion_height = 150 + 90 + (source_id - 1) * 50\n",
    "        elif answer_type == \"Chat on projects with sources and summaries\":\n",
    "            accordion_height = 500 + 90 + (source_id - 1) * 50\n",
    "        else:\n",
    "            accordion_height = 0\n",
    "\n",
    "    except Exception as e:\n",
    "        handle_error(e)\n",
    "\n",
    "    return html, accordion_height\n",
    "\n",
    "\n",
    "def get_prompt(is_bullet_point_answer):\n",
    "    if is_bullet_point_answer:\n",
    "        system_template = \"\"\" \n",
    "        Answer the question based on the context below. If the question cannot be answered using the information provided, answer with \"I am sorry. I cannot answer your question based on the provided context.\". Do not try to make up an answer. Answer the question into bullet point list.\n",
    "\n",
    "        \n",
    "        #####Start of Context#####\n",
    "        {context}\n",
    "        #####End of Context#####\n",
    "        \"\"\"\n",
    "    else:\n",
    "        system_template = \"\"\" \n",
    "        Answer the question based on the context below. If the question cannot be answered using the information provided, answer with \"I am sorry. I cannot answer your question based on the provided context.\". Do not try to make up an answer.\n",
    "\n",
    "        \n",
    "        #####Start of Context#####\n",
    "        {context}\n",
    "        #####End of Context#####\n",
    "        \"\"\"\n",
    "\n",
    "    user_template = \"Question:{question}\"\n",
    "\n",
    "    messages = [\n",
    "        SystemMessagePromptTemplate.from_template(system_template),\n",
    "        HumanMessagePromptTemplate.from_template(user_template),\n",
    "    ]\n",
    "\n",
    "    prompt = ChatPromptTemplate.from_messages(messages)\n",
    "\n",
    "    return prompt\n",
    "\n",
    "\n",
    "try:\n",
    "    st.set_page_config(page_title=\"Healthcare-related Problem Statements - CHISEL, IMDA, IdeAble.sg\",\n",
    "                       page_icon=\":robot_face:\")\n",
    "\n",
    "    st.markdown(\"<h1 style='text-align: center;'>Healthcare-related Problem Statements 😬</h1>\",\n",
    "                unsafe_allow_html=True)\n",
    "\n",
    "    import os\n",
    "    try:\n",
    "        problem_statement_pinecone_api_key = os.environ['problem_statement_pinecone_api_key']\n",
    "        problem_statement_pinecone_environment = os.environ['problem_statement_pinecone_environment']\n",
    "        problem_statement_index_name = os.environ['problem_statement_index_name']\n",
    "    except KeyError as e:\n",
    "        print(f\"Error: Missing environment variable: {e}\")\n",
    "\n",
    "    import os\n",
    "    import pinecone\n",
    "\n",
    "    try:\n",
    "        problem_statement_pinecone_api_key = os.environ['problem_statement_pinecone_api_key']\n",
    "        problem_statement_pinecone_environment = os.environ['problem_statement_pinecone_environment']\n",
    "    except KeyError as e:\n",
    "        print(f\"Error: Missing environment variable: {e}\")\n",
    "\n",
    "    if 'problem_statement_index_name' in locals():\n",
    "        problem_statement_index = pinecone.Index(problem_statement_index_name)\n",
    "        time.sleep(1)\n",
    "\n",
    "        problem_statement_vectorstore = Pinecone(\n",
    "            problem_statement_index, embed.embed_query, text_field\n",
    "        )\n",
    "    else:\n",
    "        print(\"Error: Index name not provided.\")\n",
    "\n",
    "    import os\n",
    "    import pinecone\n",
    "\n",
    "    try:\n",
    "        child_pinecone_api_key = os.environ['child_pinecone_api_key']\n",
    "        child_pinecone_environment = os.environ['child_pinecone_environment']\n",
    "        child_index_name = os.environ['child_index_name']\n",
    "    except KeyError as e:\n",
    "        print(f\"Error: Missing environment variable: {e}\")\n",
    "\n",
    "    if all(key in locals() for key in ['68636eff-3870-49b8-9f7f-799d1f82d468', 'us-east-1', 'child-serverless']):\n",
    "       pinecone.init(api_key=child_pinecone_api_key, environment=child_pinecone_environment)\n",
    "       child_index = pinecone.Index(child_index_name)\n",
    "       time.sleep(1)\n",
    "\n",
    "       child_vectorstore = Pinecone(child_index, embed.embed_query, text_field)\n",
    "    else:\n",
    "        print(\"Error: Missing one or more environment variables for the CHILD Pinecone.io Database.\")\n",
    "\n",
    "    if 'generated' not in st.session_state:\n",
    "        st.session_state['generated'] = []\n",
    "    if 'past' not in st.session_state:\n",
    "        st.session_state['past'] = []\n",
    "    if 'messages' not in st.session_state:\n",
    "        st.session_state['messages'] = [\n",
    "            {\"role\": \"system\", \"content\": \"You are a helpful assistant.\"}\n",
    "        ]\n",
    "    if 'model_name' not in st.session_state:\n",
    "        st.session_state['model_name'] = []\n",
    "    if 'problem_statement_list' not in st.session_state:\n",
    "        st.session_state['problem_statement_list'] = []\n",
    "    if 'child_response' not in st.session_state:\n",
    "        st.session_state['child_response'] = []\n",
    "    if 'accordion_html_code' not in st.session_state:\n",
    "        st.session_state['accordion_html_code'] = []\n",
    "    if 'accordion_height' not in st.session_state:\n",
    "        st.session_state['accordion_height'] = []\n",
    "    if 'market_solutions_response' not in st.session_state:\n",
    "        st.session_state['market_solutions_response'] = []\n",
    "    if 'market_solutions_sources' not in st.session_state:\n",
    "        st.session_state['market_solutions_sources'] = []\n",
    "\n",
    "    counter_placeholder = st.sidebar.empty()\n",
    "\n",
    "    st.caption(\n",
    "            \"Database currently down for migration to serverless hosting. Sorry for the inconvenience. Model: gpt-3.5-turbo / gpt-4o.\")\n",
    "\n",
    "    model_name = st.sidebar.radio(\"Choose a model:\", (\"GPT-3.5\", \"GPT-4o\"))\n",
    "\n",
    "    if model_name == \"GPT-3.5\":\n",
    "        model = \"gpt-3.5-turbo\"\n",
    "    else:\n",
    "        model = \"gpt-4o\"\n",
    "\n",
    "    OPENAI_API_KEY = \"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\"\n",
    "\n",
    "    llm = ChatOpenAI(\n",
    "        temperature=0,\n",
    "        openai_api_key=OPENAI_API_KEY,\n",
    "        model_name=model,\n",
    "        verbose=False,\n",
    "    )\n",
    "\n",
    "    def generate_response(prompt):\n",
    "        query = prompt\n",
    "        qa_prompt = get_prompt(False)\n",
    "        st.session_state['messages'].append(\n",
    "            {\"role\": \"user\", \"content\": prompt})\n",
    "\n",
    "        ps_query = query\n",
    "\n",
    "        docs_and_scores = problem_statement_vectorstore.similarity_search_with_score(\n",
    "            ps_query)\n",
    "\n",
    "        raw_problem_statement_list = []\n",
    "        problem_statement_list = []\n",
    "        for doc in docs_and_scores:\n",
    "            if 'year' in list(doc)[0].metadata:\n",
    "                year = list(doc)[0].metadata['year']\n",
    "            else:\n",
    "                year = 0\n",
    "\n",
    "            if 'category' in list(doc)[0].metadata:\n",
    "                category = list(doc)[0].metadata['category']\n",
    "            else:\n",
    "                category = \"\"\n",
    "\n",
    "            if 'requestor' in list(doc)[0].metadata:\n",
    "                requestor = list(doc)[0].metadata['requestor']\n",
    "            else:\n",
    "                requestor = \"\"\n",
    "\n",
    "            if 'problem_statement' in list(doc)[0].metadata:\n",
    "                problem_statement = list(doc)[0].metadata['problem_statement']\n",
    "            else:\n",
    "                problem_statement = \"\"\n",
    "\n",
    "            if 'contributor' in list(doc)[0].metadata:\n",
    "                contributor = list(\n",
    "                    doc)[0].metadata['contributor']\n",
    "            else:\n",
    "                contributor = \"\"\n",
    "\n",
    "            if 'background' in list(doc)[0].metadata:\n",
    "                background = list(\n",
    "                    doc)[0].metadata['background']\n",
    "            else:\n",
    "                background = \"\"\n",
    "\n",
    "            if 'desired_outcomes' in list(doc)[0].metadata:\n",
    "                desired_outcomes = list(\n",
    "                    doc)[0].metadata['desired_outcomes']\n",
    "            else:\n",
    "                desired_outcomes = \"\"\n",
    "\n",
    "            if 'funding' in list(doc)[0].metadata:\n",
    "                funding = list(\n",
    "                    doc)[0].metadata['funding']\n",
    "            else:\n",
    "                funding = \"\"\n",
    "\n",
    "            raw_problem_statement = problem_statement.strip()\n",
    "            raw_problem_statement = raw_problem_statement.lower()\n",
    "            raw_problem_statement = raw_problem_statement.replace(\" \", \"\")\n",
    "            raw_problem_statement = raw_problem_statement.replace(\"\\n\", \"\")\n",
    "\n",
    "            if raw_problem_statement not in raw_problem_statement_list:\n",
    "                raw_problem_statement_list.append(raw_problem_statement)\n",
    "\n",
    "                score = list(doc)[1]\n",
    "                score = float(score)\n",
    "                score = score * 100\n",
    "\n",
    "                if score >= 80:\n",
    "                    score = str(round(score, 2))\n",
    "                    problem_statement_list.append(\n",
    "                        {\"score\": score, \"problem_statement\": problem_statement, \"year\": year, \"category\": category, \"requestor\": requestor, \"contributor\": contributor, \"background\": background, \"desired_outcomes\": desired_outcomes, \"funding\": funding, })\n",
    "\n",
    "        max_token_limit = 4096\n",
    "\n",
    "        child_query = query\n",
    "\n",
    "        qa_prompt = get_prompt(True)\n",
    "\n",
    "        child_memory = ConversationSummaryBufferMemory(\n",
    "            llm=llm,\n",
    "            memory_key=\"chat_history\",\n",
    "            return_messages=True,\n",
    "            max_token_limit=max_token_limit,\n",
    "            input_key='question',\n",
    "            output_key='answer',\n",
    "        )\n",
    "\n",
    "        chain = ConversationalRetrievalChain.from_llm(\n",
    "            llm,\n",
    "            retriever=child_vectorstore.as_retriever(),\n",
    "            combine_docs_chain_kwargs={\"prompt\": qa_prompt},\n",
    "            memory=child_memory,\n",
    "            return_source_documents=True,\n",
    "            verbose=False,\n",
    "        )\n",
    "\n",
    "        result = chain({'question': child_query})\n",
    "        raw_answer = result['answer']\n",
    "        child_response = sanitize_answer(raw_answer)\n",
    "\n",
    "        child_sources = \"\"\n",
    "\n",
    "        if \"context provided does not\" in child_response.lower():\n",
    "            pass\n",
    "        elif \"cannot answer your question\" in child_response.lower():\n",
    "            pass\n",
    "        else:\n",
    "            source_documents = result['source_documents']\n",
    "            source_list = []\n",
    "            for source_document in source_documents:\n",
    "                source = source_document.metadata['source']\n",
    "                source_list.append(source)\n",
    "\n",
    "            source_list = list(set(source_list))\n",
    "            child_sources = ', '.join(source_list)\n",
    "\n",
    "        st.session_state['messages'].append(\n",
    "            {\"role\": \"assistant\", \"content\": problem_statement})\n",
    "\n",
    "        return problem_statement_list, child_response, child_sources\n",
    "\n",
    "    response_container = st.container()\n",
    "    container = st.container()\n",
    "\n",
    "    with container:\n",
    "        with st.form(key='my_form', clear_on_submit=True):\n",
    "            user_input = st.text_area(\"Prompt: Problem Statement\", key='input', height=50)\n",
    "            submit_button = st.form_submit_button(label='Send')\n",
    "\n",
    "        import pinecone\n",
    "\n",
    "        problem_statement_index_name = \"problem-statements-ttsh\"\n",
    "        problem_statement_pinecone_api_key=\"a7f95b87-bb0a-4202-b2f0-5ea2d682dc78\"\n",
    "        problem_statement_pinecone_environment = \"us-east-1\"\n",
    "\n",
    "        import os\n",
    "        from pinecone import Pinecone, ServerlessSpec\n",
    "\n",
    "        problem_statement_index_name = \"problem-statements-ttsh\"\n",
    "\n",
    "        pc = Pinecone(\n",
    "            api_key=problem_statement_pinecone_api_key,\n",
    "            environment=problem_statement_pinecone_environment\n",
    "        )\n",
    "\n",
    "        if problem_statement_index_name.lower() not in pc.list_indexes().names():\n",
    "            pc.create_index(\n",
    "                name=problem_statement_index_name.lower(),\n",
    "                dimension=1536,\n",
    "                metric='cosine',\n",
    "                spec=ServerlessSpec(\n",
    "                    cloud='aws',\n",
    "                    region='us-west-1'\n",
    "                )\n",
    "           )\n",
    "\n",
    "        import pinecone\n",
    "        import faiss\n",
    "        import numpy as np\n",
    "        import streamlit as st\n",
    "\n",
    "        problem_statement_index_name = \"problem-statements-ttsh\"\n",
    "\n",
    "        pc = pinecone.Pinecone(api_key=problem_statement_pinecone_api_key, environment=problem_statement_pinecone_environment)\n",
    "\n",
    "        dimension = 1536\n",
    "        index = faiss.IndexFlatL2(dimension)\n",
    "\n",
    "        vectors = np.random.random((100, dimension)).astype('float32')\n",
    "        index.add(vectors)\n",
    "\n",
    "        def generate_response(user_input):\n",
    "            query_vector = np.random.random((1, dimension)).astype('float32')\n",
    "            k = 5\n",
    "            distances, indices = index.search(query_vector, k)\n",
    "    \n",
    "            problem_statement_list = indices.tolist()\n",
    "            child_response = \"Generated response based on input\"\n",
    "            child_sources = \"List of sources\"\n",
    "            return problem_statement_list, child_response, child_sources\n",
    "\n",
    "        if submit_button and user_input:\n",
    "            problem_statement_list, child_response, child_sources = generate_response(user_input)\n",
    "            st.session_state['past'].append(user_input)\n",
    "            st.session_state['generated'].append(\"\")\n",
    "            st.session_state['model_name'].append(model_name)\n",
    "            st.session_state['problem_statement_list'].append(problem_statement_list)\n",
    "            st.session_state['child_response'].append(child_response)\n",
    "\n",
    "            accordion_html_code = \"\"\n",
    "            accordion_height = 0\n",
    "            sources = child_sources.strip()\n",
    "         \n",
    "            if len(sources) >= 5:\n",
    "                query = user_input\n",
    "                answer_type = \"Chat on projects with sources and summaries\"\n",
    "                accordion_html_code, accordion_height = accordion(\n",
    "                                    query, sources, answer_type)\n",
    "                accordion_html_code = str(accordion_html_code)\n",
    "\n",
    "            st.session_state['accordion_html_code'].append(accordion_html_code)\n",
    "            st.session_state['accordion_height'].append(accordion_height)\n",
    "\n",
    "    if st.session_state['generated']:\n",
    "        with st.container():\n",
    "            for i in range(len(st.session_state['generated'])):\n",
    "                st.message(st.session_state[\"past\"][i], is_user=True, key=str(i) + '_user')\n",
    "\n",
    "                if len(st.session_state[\"problem_statement_list\"][i]) < 1:\n",
    "                    st.markdown(\n",
    "                        f\"\"\"<span style=\"word-wrap:break-word;\">No similar problem statement found in the system. Do you want to submit a new problem statement?</span>\"\"\",\n",
    "                        unsafe_allow_html=True)\n",
    "                    st.markdown(\n",
    "                        f\"\"\"<span style=\"word-wrap:break-word;\"><a href=\"mailto:chisel@chi.sg\" target=\"_blank\">Mail to chisel@chi.sg to submit a new problem statement</a></span>\"\"\",\n",
    "                        unsafe_allow_html=True)\n",
    "                elif len(st.session_state[\"problem_statement_list\"][i]) == 1:\n",
    "                    for problem_statement_data in st.session_state[\"problem_statement_list\"][i]:\n",
    "                        score = problem_statement_data[\"score\"]\n",
    "                        year = problem_statement_data[\"year\"]\n",
    "                        category = problem_statement_data[\"category\"]\n",
    "                        requestor = problem_statement_data[\"requestor\"].strip().replace('\\n', '<br>')\n",
    "                        problem_statement = problem_statement_data[\"problem_statement\"]\n",
    "                        contributor = problem_statement_data[\"contributor\"]\n",
    "                        background = problem_statement_data[\"background\"]\n",
    "                        desired_outcomes = problem_statement_data[\"desired_outcomes\"]\n",
    "                        funding = problem_statement_data[\"funding\"]\n",
    "\n",
    "                        st.markdown(\n",
    "                            f\"\"\"<span style=\"word-wrap:break-word;\"><strong>Problem Statement Found:</strong> {problem_statement}</span> <span style=\"word-wrap:break-word; font-style: italic;\">(Relevance Score: {score}%)</span>\"\"\",\n",
    "                            unsafe_allow_html=True)\n",
    "                        st.markdown(f\"\"\"<span style=\"word-wrap:break-word;\"><strong>Year:</strong> {year}</span>\"\"\",\n",
    "                                    unsafe_allow_html=True)\n",
    "                        st.markdown(\n",
    "                            f\"\"\"<span style=\"word-wrap:break-word;\"><strong>Requestor/Dept/Institution:</strong><br>{requestor}</span>\"\"\",\n",
    "                            unsafe_allow_html=True)\n",
    "                        st.markdown(\n",
    "                            f\"\"\"<span style=\"word-wrap:break-word;\"><strong>Contributor:</strong><br>{contributor}</span>\"\"\",\n",
    "                            unsafe_allow_html=True)\n",
    "\n",
    "                        with st.expander(\"See more\"):\n",
    "                            st.markdown(f\"\"\"<span style=\"word-wrap:break-word;\"><strong>Category:</strong> {category}</span>\"\"\",\n",
    "                                        unsafe_allow_html=True)\n",
    "                            st.markdown(f\"\"\"<span style=\"word-wrap:break-word;\"><strong>Background:</strong> {background}</span>\"\"\",\n",
    "                                        unsafe_allow_html=True)\n",
    "                            st.markdown(\n",
    "                                f\"\"\"<span style=\"word-wrap:break-word;\"><strong>Desired Outcomes:</strong> {desired_outcomes}</span>\"\"\",\n",
    "                                unsafe_allow_html=True)\n",
    "                            st.markdown(f\"\"\"<span style=\"word-wrap:break-word;\"><strong>Funding:</strong> {funding}</span>\"\"\",\n",
    "                                        unsafe_allow_html=True)\n",
    "\n",
    "                        st.markdown(f\"\"\"<br>\"\"\", unsafe_allow_html=True)\n",
    "\n",
    "                else:\n",
    "                    for counter, problem_statement_data in enumerate(st.session_state[\"problem_statement_list\"][i], 1):\n",
    "                        score = problem_statement_data[\"score\"]\n",
    "                        year = problem_statement_data[\"year\"]\n",
    "                        requestor = problem_statement_data[\"requestor\"].strip().replace('\\n', '<br>')\n",
    "                        problem_statement = problem_statement_data[\"problem_statement\"]\n",
    "                        contributor = problem_statement_data[\"contributor\"]\n",
    "                        background = problem_statement_data[\"background\"]\n",
    "                        desired_outcomes = problem_statement_data[\"desired_outcomes\"]\n",
    "                        funding = problem_statement_data[\"funding\"]\n",
    "\n",
    "                        st.markdown(\n",
    "                            f\"\"\"<span style=\"word-wrap:break-word;\"><strong>Problem Statement Found {counter}:</strong> {problem_statement}</span> <span style=\"word-wrap:break-word; font-style: italic;\">(Relevance Score: {score}%)</span>\"\"\",\n",
    "                            unsafe_allow_html=True)\n",
    "                        st.markdown(f\"\"\"<span style=\"word-wrap:break-word;\"><strong>Year:</strong> {year}</span>\"\"\",\n",
    "                                    unsafe_allow_html=True)\n",
    "                        st.markdown(\n",
    "                            f\"\"\"<span style=\"word-wrap:break-word;\"><strong>Requestor/Dept/Institution:</strong><br>{requestor}</span>\"\"\",\n",
    "                            unsafe_allow_html=True)\n",
    "                        st.markdown(\n",
    "                            f\"\"\"<span style=\"word-wrap:break-word;\"><strong>Contributor:</strong><br>{contributor}</span>\"\"\",\n",
    "                            unsafe_allow_html=True)\n",
    "\n",
    "                        with st.expander(\"See more\"):\n",
    "                            st.markdown(f\"\"\"<span style=\"word-wrap:break-word;\"><strong>Background:</strong> {background}</span>\"\"\",\n",
    "                                        unsafe_allow_html=True)\n",
    "                            st.markdown(\n",
    "                                f\"\"\"<span style=\"word-wrap:break-word;\"><strong>Desired Outcomes:</strong> {desired_outcomes}</span>\"\"\",\n",
    "                                unsafe_allow_html=True)\n",
    "                            st.markdown(f\"\"\"<span style=\"word-wrap:break-word;\"><strong>Funding:</strong> {funding}</span>\"\"\",\n",
    "                                        unsafe_allow_html=True)\n",
    "\n",
    "                        st.markdown(f\"\"\"<br>\"\"\", unsafe_allow_html=True)\n",
    "\n",
    "                st.message(f'Similar projects found in CHILD: {st.session_state[\"child_response\"][i]}', key=str(i))\n",
    "\n",
    "                accordion_html_code = st.session_state[\"accordion_html_code\"][i]\n",
    "                accordion_height = st.session_state[\"accordion_height\"][i]\n",
    "\n",
    "                if accordion_height > 0:\n",
    "                    st.components.v1.html(accordion_html_code, height=accordion_height)\n",
    "\n",
    "except Exception as e:\n",
    "    handle_error(e)\n",
    "\n",
    "def handle_error(e):\n",
    "    error_message = ''\n",
    "    st.error('An error has occurred. Please try again.', icon=\"🚨\")\n",
    "\n",
    "    if hasattr(e, 'message'):\n",
    "        error_message = e.message\n",
    "    else:\n",
    "        error_message = str(e)\n",
    "\n",
    "    st.error('ERROR MESSAGE: {}'.format(error_message), icon=\"🚨\")\n",
    "\n",
    "    exc_type, exc_obj, exc_tb = sys.exc_info()\n",
    "    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]\n",
    "    st.error(f'Error Type: {exc_type}', icon=\"🚨\")\n",
    "    st.error(f'File Name: {fname}', icon=\"🚨\")\n",
    "    st.error(f'Line Number: {exc_tb.tb_lineno}', icon=\"🚨\")\n",
    "    print(traceback.format_exc())\n",
    "\n",
    "print('We appreciate the opportunity to serve you')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

