import os
# from pinecone import Pinecone, ServerlessSpec
import pinecone
import sys
import time
import traceback
import openai
import streamlit as st
from streamlit_chat import message
import streamlit.components.v1 as components
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

try:
    import environment_variables
except ImportError:
    pass

element_id = 0


def sanitize_answer(raw_answer):
    answer = ''
    chunks = raw_answer.split('\n')
    sub_string_list = []
    sub_string_list.append('- SOURCE:')
    sub_string_list.append('- Source:')
    sub_string_list.append('- source:')
    sub_string_list.append('- SOURCES:')
    sub_string_list.append('- Sources:')
    sub_string_list.append('- sources:')
    sub_string_list.append('(SOURCE:')
    sub_string_list.append('(Source:')
    sub_string_list.append('(source:')
    sub_string_list.append('(SOURCES:')
    sub_string_list.append('(Sources:')
    sub_string_list.append('(sources:')
    sub_string_list.append('SOURCE:')
    sub_string_list.append('Source:')
    sub_string_list.append('source:')
    sub_string_list.append('SOURCES:')
    sub_string_list.append('Sources:')
    sub_string_list.append('sources:')

    try:
        for chunk in chunks:
            temp_string = str(chunk)
            temp_string = temp_string.strip()
            temp_string_lowercase = temp_string.lower()
            answer_text = ''

            if temp_string_lowercase.find('source') != -1:
                for sub_string in sub_string_list:
                    if temp_string.find(sub_string) != -1:
                        # print(f'{sub_string} - {temp_string}')
                        temp_string = temp_string[:temp_string.index(
                            sub_string)]

            # Append answer text
            answer_text = temp_string.strip()
            if len(answer_text) > 0:
                # print(answer_text)
                answer = answer + '\n\n' + answer_text

        answer = answer.strip()
    except Exception as e:
        error_message = ''
        # st.text('Hello World')
        st.error('An error has occurred. Please try again.', icon="🚨")
        # Just print(e) is cleaner and more likely what you want,
        # but if you insist on printing message specifically whenever possible...
        if hasattr(e, 'message'):
            error_message = e.message
        else:
            error_message = e
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        st.error('ERROR MESSAGE: {}'.format(error_message))

    return answer


def get_source_list(sources):
    chunks = []
    source_list = []
    allowed_file_extensions = ['.pdf']

    try:
        if '\n' in sources:
            chunks = sources.split('\n')
        elif ',' in sources:
            chunks = sources.split(',')
        else:
            source = sources.strip()
            chunks.append(source)
            # source_list.append(source)

        for chunk in chunks:
            chunk = chunk.strip()
            if (len(chunk) >= 5):
                first_two_chars = chunk[:2]
                # print('first_two_chars: {}'.format(first_two_chars))
                if first_two_chars == '- ':
                    chunk = chunk[2:]
                extension = os.path.splitext(chunk)[1]
                extension = extension.lower()
                # print('extension: {}'.format(extension))

                if extension in allowed_file_extensions:
                    # print('{} is in {}'.format(extension, allowed_file_extensions))
                    source_list.append(chunk)
    except Exception as e:
        error_message = ''
        # st.text('Hello World')
        st.error('An error has occurred. Please try again.', icon="🚨")
        # Just print(e) is cleaner and more likely what you want,
        # but if you insist on printing message specifically whenever possible...
        if hasattr(e, 'message'):
            error_message = e.message
        else:
            error_message = e
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        st.error('ERROR MESSAGE: {}'.format(error_message))

    return source_list


def accordion(query, sources, answer_type):
    html = ''
    global element_id
    sources = sources.strip()
    source_id = 0
    accordion_height = 0
    # 5 = 290
    # 4 = 240
    # 3 = 190
    # 2 = 140
    # 1 =  90

    # st.sidebar.text(query)

    try:
        if (len(sources)) < 5:
            return html, accordion_height
        else:
            source_list = get_source_list(sources)

            if answer_type == "Chat on projects with sources and summaries":
                model_name = 'text-embedding-ada-002'

                embed = OpenAIEmbeddings(
                    model=model_name,
                    openai_api_key=OPENAI_API_KEY
                )

                text_field = "text"
                # index_name = "langchain-ttsh"
                # Switch back to normal index for langchain
                # index = pinecone.Index(index_name)

                vectorstore = Pinecone(
                    child_index, embed.embed_query, text_field
                )

                # Completion llm
                llm = ChatOpenAI(
                    openai_api_key=OPENAI_API_KEY,
                    # model_name='gpt-3.5-turbo',
                    model_name=model,
                    temperature=0.0,
                    verbose=False,
                )

                # st.sidebar.text('Connected to Pinecone....')
                # st.sidebar.text('================================')

        html = html + '<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet">'
        html = html + '<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js"></script>'
        html = html + '<div class="m-4">'
        html = html + '<div class="accordion" id="myAccordion">'

        for source in source_list:
            source = source.strip()
            source_id = source_id + 1
            element_id = element_id + 1
            # heading_number =

            if answer_type == "Chat on projects with sources":
                summary = ""
            elif answer_type == "Chat on projects with sources and summaries":
                summary_text = ""
                # summary_text = query
                summary = ""

                # xq = llm.encode(query).tolist()
                # result = index.query(xq, top_k=10, filter={'source': source})
                # st.sidebar.text(result)
                # st.sidebar.text('===========================================')

                # Create a retriever
                retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 9, 'filter': {'source': source}})
                docs = retriever.get_relevant_documents(query)
                # st.sidebar.text(f'Number of retrieved docs = {len(docs)}')
                # st.sidebar.text(docs)
                # st.sidebar.text('===========================================')

                # qw_changed_to_comment chain = load_summarize_chain(llm, chain_type="stuff")
                # search = vectordb.similarity_search(" ")
                # summary = chain.run(input_documents=search, question="Write a summary within 200 words.")
                # qw_changed_to_comment summary_text = chain.run(docs)

                # qw_insertCode_begin
                prompt_template_summ = """Write a concise summary of the following, be sure to include the aims, lessons learnt and conclusions: "{text_sum}"
                ALWAYS give your answer in point form like in the example.
                
                Example:
                Aim: This project aimed to determine if videoconsult was superior to face to face consults \n

                
                Lessons Learnt: None found \n

                
                Conclusions: Patients preferred videoconsults to physical consults.
                
                
                #CONCISE SUMMARY:"""
                prompt_summ = PromptTemplate.from_template(
                    prompt_template_summ)
                llm_chain_summ = LLMChain(llm=llm, prompt=prompt_summ)
                stuff_chain = StuffDocumentsChain(
                    llm_chain=llm_chain_summ, document_variable_name="text_sum")
                summary_text = stuff_chain.run(docs)
                # qw_insertCode_end

                # qw_edit: summary_text = summary_text.strip()
                summary_text = summary_text.strip().replace("\n", "<br>")
                if len(summary_text) > 0:
                    # summary = " <br><strong>Summary:</strong> Work in progress for summary"
                    summary = f" <br><strong>Summary:</strong> <br> {summary_text}"
            else:
                summary = ""

            html = html + '<div class="accordion-item">'
            html = html + \
                f'<h2 class="accordion-header" id="heading-{source_id}">'
            html = html + \
                f'<button type="button" class="accordion-button collapsed" data-bs-toggle="collapse" data-bs-target="#collapse-{source_id}">Source {source_id}</button>'
            html = html + '</h2>'
            html = html + \
                f'<div id="collapse-{source_id}" class="accordion-collapse collapse" data-bs-parent="#myAccordion">'
            html = html + '<div class="card-body">'
            html = html + \
                f'<p><strong><a href="https://child.chi.sg/files/pdffiles/{source}" target="_blank"><span style="overflow-wrap: break-word;">{source}</span></a></strong>{summary}</p>'
            html = html + '</div>'
            html = html + '</div>'
            html = html + '</div>'

        html = html + '</div>'
        html = html + '</div>'

        html = f"""
                {html}
                """

        # html = str(html)
        # accordion_height = 75 + 90 + (source_id - 1) * 50
        # accordion_height = 200 + 90 + (source_id - 1) * 50
        if answer_type == "Chat on projects with sources":
            accordion_height = 150 + 90 + (source_id - 1) * 50
        elif answer_type == "Chat on projects with sources and summaries":
            accordion_height = 500 + 90 + (source_id - 1) * 50
        else:
            accordion_height = 0

    except Exception as e:
        error_message = ''
        # st.text('Hello World')
        st.error('An error has occurred. Please try again.', icon="🚨")
        # Just print(e) is cleaner and more likely what you want,
        # but if you insist on printing message specifically whenever possible...
        if hasattr(e, 'message'):
            error_message = e.message
        else:
            error_message = e
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        st.error('ERROR MESSAGE: {}'.format(error_message))

    return html, accordion_height


def get_prompt(is_bullet_point_answer):
    # Define the system message template
    if is_bullet_point_answer == True:
        system_template = """ 
        Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I am sorry. I cannot answer your question based on the provided context.". Do not try to make up an answer. Answer the question into bullet point list.

        
        #####Start of Context#####
        {context}
        #####End of Context#####
        """
    else:
        system_template = """ 
        Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I am sorry. I cannot answer your question based on the provided context.". Do not try to make up an answer.

        
        #####Start of Context#####
        {context}
        #####End of Context#####
        """

    user_template = "Question:```{question}```"

    # Create the chat prompt templates
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(user_template),
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    return prompt


try:
    # Setting page title and header

    
        
    st.set_page_config(page_title="Healthcare-related Problem Statements - CHISEL, IMDA, IdeAble.sg",
                       page_icon=":robot_face:")
    

    st.markdown("<h1 style='text-align: center;'>Healthcare-related Problem Statements 😬</h1>",
                unsafe_allow_html=True)

    # st.image('./banner_psms.jpg')

    


    # Step 2: Get Pinecone.io database specific environment variables

    import os
    try:
        problem_statement_pinecone_api_key = os.environ['problem_statement_pinecone_api_key']
        problem_statement_pinecone_environment = os.environ['problem_statement_pinecone_environment']
        problem_statement_index_name = os.environ['problem_statement_index_name']
    except KeyError as e:
         print(f"Error: Missing environment variable: {e}")

    # Initialize connection to pinecone (get API key at app.pinecone.io)
    
    import os
    import pinecone

    # Load environment variables
    try:
        problem_statement_pinecone_api_key = os.environ['problem_statement_pinecone_api_key']
        problem_statement_pinecone_environment = os.environ['problem_statement_pinecone_environment']
    except KeyError as e:
        print(f"Error: Missing environment variable: {e}")

    if 'problem_statement_index_name' in locals():
        # Connect to the index
        problem_statement_index = pinecone.Index(problem_statement_index_name)
        # Wait a moment for the index to be fully initialized
        time.sleep(1)

        problem_statement_vectorstore = Pinecone(
            problem_statement_index, embed.embed_query, text_field
        )
    else:
        print("Error: Index name not provided.")



    # ==================================================== #

    import os
    import pinecone

    # Load environment variables for CHILD Pinecone.io Database
    try:
        child_pinecone_api_key = os.environ['child_pinecone_api_key']
        child_pinecone_environment = os.environ['child_pinecone_environment']
        child_index_name = os.environ['child_index_name']
    except KeyError as e:
        print(f"Error: Missing environment variable: {e}")

    if all(key in locals() for key in ['68636eff-3870-49b8-9f7f-799d1f82d468', 'us-east-1', 'child-serverless']):
       pinecone.init(api_key=child_pinecone_api_key, environment=child_pinecone_environment)
       child_index = pinecone.Index(child_index_name)
       time.sleep(1)

       # Initialize the Pinecone object for the CHILD database
       child_vectorstore = Pinecone(child_index, embed.embed_query, text_field)
    else:
        print("Error: Missing one or more environment variables for the CHILD Pinecone.io Database.")
    # ==================================================== #

    # Initialise session state variables
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    if 'model_name' not in st.session_state:
        st.session_state['model_name'] = []
    if 'problem_statement_list' not in st.session_state:
        st.session_state['problem_statement_list'] = []
    if 'child_response' not in st.session_state:
        st.session_state['child_response'] = []
    if 'accordion_html_code' not in st.session_state:
        st.session_state['accordion_html_code'] = []
    if 'accordion_height' not in st.session_state:
        st.session_state['accordion_height'] = []
    if 'market_solutions_response' not in st.session_state:
        st.session_state['market_solutions_response'] = []
    if 'market_solutions_sources' not in st.session_state:
        st.session_state['market_solutions_sources'] = []

    # Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
    # st.sidebar.title("Sidebar")
    counter_placeholder = st.sidebar.empty()
    # counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
    # clear_button = st.sidebar.button("Clear Conversation", key="clear")

    # st.sidebar.markdown(
    #    '1. <a href="https://child-projects.streamlit.app/" target="_blank">CHILD Project Collection</a>', unsafe_allow_html=True)
    # st.sidebar.markdown(
    #     '2. <a href="https://chisel-ps-uploader.streamlit.app/" target="_blank">Submit a New Problem Statement</a>', unsafe_allow_html=True)
    # st.sidebar.markdown(
    #    '2. <a href="https://chisel-psms.streamlit.app/" target="_blank">Problem Statements to Market Solutions</a>', unsafe_allow_html=True)
    # st.sidebar.markdown(
    #     '4. <a href="https://chisel-ms-uploader.streamlit.app/" target="_blank">Submit a New Market Solution</a>', unsafe_allow_html=True)

    # with st.sidebar:
    st.caption(
            "Database currently down for migration to serverless hosting. Sorry for the inconvenience. Model: gpt-3.5-turbo / gpt-4o.")

    model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4o"))
    # model_name = "GPT-4"
    # Map model names to OpenAI model IDs
    if model_name == "GPT-3.5":
        model = "gpt-3.5-turbo"
    else:
        model = "gpt-4o"
        # model = "gpt-4-1106-preview"
        # model = "gpt-4-turbo"

    # Define your OpenAI API key
    OPENAI_API_KEY = "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"

    # Initialize the large language model
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
        model_name=model,
        verbose=False,
    )


    # reset everything
    # if clear_button:
    # st.session_state['generated'] = []
    # st.session_state['past'] = []
    # st.session_state['messages'] = [
    #     {"role": "system", "content": "You are a helpful assistant."}
    # ]
    # st.session_state['model_name'] = []
    # counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")

    # generate a response

    def generate_response(prompt):
        query = prompt
        qa_prompt = get_prompt(False)
        st.session_state['messages'].append(
            {"role": "user", "content": prompt})

        ######################################################
        ps_query = query
        # docs = vectorstore.similarity_search(
        #     query,  # our search query
        #     k=3,  # return 3 most relevant docs
        #     # include_metadata=True
        # )

        # for doc in docs:
        #     st.sidebar.text(doc.metadata['problem_statement'])

        docs_and_scores = problem_statement_vectorstore.similarity_search_with_score(
            ps_query)

        raw_problem_statement_list = []
        problem_statement_list = []
        for doc in docs_and_scores:
            if 'year' in list(doc)[0].metadata:
                year = list(doc)[0].metadata['year']
            else:
                year = 0

            if 'category' in list(doc)[0].metadata:
                category = list(doc)[0].metadata['category']
            else:
                category = ""

            if 'requestor' in list(doc)[0].metadata:
                requestor = list(doc)[0].metadata['requestor']
            else:
                requestor = ""

            if 'problem_statement' in list(doc)[0].metadata:
                problem_statement = list(doc)[0].metadata['problem_statement']
            else:
                problem_statement = ""

            if 'contributor' in list(doc)[0].metadata:
                contributor = list(
                    doc)[0].metadata['contributor']
            else:
                contributor = ""

            if 'background' in list(doc)[0].metadata:
                background = list(
                    doc)[0].metadata['background']
            else:
                background = ""

            if 'desired_outcomes' in list(doc)[0].metadata:
                desired_outcomes = list(
                    doc)[0].metadata['desired_outcomes']
            else:
                desired_outcomes = ""

            if 'funding' in list(doc)[0].metadata:
                funding = list(
                    doc)[0].metadata['funding']
            else:
                funding = ""

            raw_problem_statement = problem_statement.strip()
            raw_problem_statement = raw_problem_statement.lower()
            raw_problem_statement = raw_problem_statement.replace(" ", "")
            raw_problem_statement = raw_problem_statement.replace("\n", "")

            if raw_problem_statement not in raw_problem_statement_list:
                raw_problem_statement_list.append(raw_problem_statement)

                # st.text(problem_statement)
                # st.sidebar.text(doc)
                score = list(doc)[1]
                score = float(score)
                score = score * 100
                # st.text(score)

                if score >= 80:
                    score = str(round(score, 2))
                    problem_statement_list.append(
                        {"score": score, "problem_statement": problem_statement, "year": year, "category": category, "requestor": requestor, "contributor": contributor, "background": background, "desired_outcomes": desired_outcomes, "funding": funding, })

        max_token_limit = 4096
        # Problem Statement
        ######################################################
        # problem_statement_memory = ConversationSummaryBufferMemory(
        #     llm=llm,
        #     memory_key="chat_history",
        #     return_messages=True,
        #     max_token_limit=max_token_limit,
        # )

        # chain = ConversationalRetrievalChain.from_llm(
        #     llm,
        #     retriever=problem_statement_vectorstore.as_retriever(),
        #     # retriever=problem_statement_vectorstore.as_retriever(search_kwargs={"k": 3}),
        #     memory=problem_statement_memory,
        #     combine_docs_chain_kwargs={"prompt": qa_prompt},
        # )
        # response = chain.run({'question': ps_query})
        # problem_statement_response = response.strip()
        # st.sidebar.text(response)
        # st.sidebar.text(problem_statement_response)
        ######################################################

        # Problem Statement
        ######################################################
        child_query = query
        # if "project on" in query.lower():
        #     pass
        # elif "projects on" in query.lower():
        #     pass
        # elif "list all projects" in query.lower():
        #     pass
        # elif "list all project" in query.lower():
        #     pass
        # elif "list projects" in query.lower():
        #     pass
        # elif "list project" in query.lower():
        #     pass
        # else:
        #     child_query = f"Projects on {query}"

        qa_prompt = get_prompt(True)

        child_memory = ConversationSummaryBufferMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=max_token_limit,
            input_key='question',
            output_key='answer',
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=child_vectorstore.as_retriever(),
            # retriever=problem_statement_vectorstore.as_retriever(search_kwargs={"k": 3}),
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            memory=child_memory,
            return_source_documents=True,
            verbose=False,
        )
        # response = chain.run({'question': query})
        # child_response = response.strip()

        # child_query = f'{child_query}. Create a bulleted list.'
        # st.sidebar.text(child_query)

        result = chain({'question': child_query})
        raw_answer = result['answer']
        child_response = sanitize_answer(raw_answer)

        child_sources = ""

        # st.sidebar.text(child_response)

        if "context provided does not" in child_response.lower():
            pass
        elif "cannot answer your question" in child_response.lower():
            pass
        else:
            source_documents = result['source_documents']
            source_list = []
            for source_document in source_documents:
                source = source_document.metadata['source']
                source_list.append(source)

            source_list = list(set(source_list))
            child_sources = ', '.join(source_list)
        ######################################################

        st.session_state['messages'].append(
            {"role": "assistant", "content": problem_statement})

        return problem_statement_list, child_response, child_sources

    # container for chat history
    response_container = st.container()
    # container for text box
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("Prompt: Problem Statement", key='input', height=50)
            submit_button = st.form_submit_button(label='Send')


        import pinecone

        # Assuming you have the index name stored in a variable
        problem_statement_index_name = "problem-statements-ttsh"
        problem_statement_pinecone_api_key="a7f95b87-bb0a-4202-b2f0-5ea2d682dc78"
        problem_statement_pinecone_environment = "us-east-1"

        

        import os
        from pinecone import Pinecone, ServerlessSpec

        # Assuming you have the index name stored in a variable
        problem_statement_index_name = "problem-statements-ttsh"

        # Initialize Pinecone
        pc = Pinecone(
            api_key=problem_statement_pinecone_api_key,
            environment=problem_statement_pinecone_environment
        )

        # Now create the index
        if problem_statement_index_name.lower() not in pc.list_indexes().names():
            pc.create_index(
                name=problem_statement_index_name.lower(),
                dimension=1536,  # Replace with your index dimension
                metric='cosine',  # Specify your desired metric
                spec=ServerlessSpec(
                    cloud='aws',  # Specify your desired cloud provider
                    region='us-west-1'  # Specify your desired region
                )
           )


        import pinecone
        import faiss
        import numpy as np
        import streamlit as st

        # Assuming `problem_statement_index_name` is defined and initialized
        problem_statement_index_name = "problem-statements-ttsh"

        # Initialize Pinecone
        pc = pinecone.Pinecone(api_key=problem_statement_pinecone_api_key, environment=problem_statement_pinecone_environment)

        # Initialize Faiss index
        dimension = 1536  # Example dimension
        index = faiss.IndexFlatL2(dimension)

        # Assuming you have some precomputed vectors
        vectors = np.random.random((100, dimension)).astype('float32')
        index.add(vectors)

        def generate_response(user_input):
            # This is a placeholder function. Implement your logic here.
            query_vector = np.random.random((1, dimension)).astype('float32')
            k = 5
            distances, indices = index.search(query_vector, k)
    
            problem_statement_list = indices.tolist()
            child_response = "Generated response based on input"
            child_sources = "List of sources"
            return problem_statement_list, child_response, child_sources


        if submit_button and user_input:
            problem_statement_list, child_response, child_sources = generate_response(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append("")
            st.session_state['model_name'].append(model_name)
            st.session_state['problem_statement_list'].append(problem_statement_list)
            st.session_state['child_response'].append(child_response)

            accordion_html_code = ""
            accordion_height = 0
            sources = child_sources.strip()
            # st.sidebar.text(f'sources: {sources}')


            if (len(sources)) >= 5:
                query = user_input
                answer_type = "Chat on projects with sources and summaries"
                accordion_html_code, accordion_height = accordion(
                    query, sources, answer_type)
                accordion_html_code = str(accordion_html_code)

            st.session_state['accordion_html_code'].append(accordion_html_code)
            st.session_state['accordion_height'].append(accordion_height)

        import streamlit as st
        import sys
        import os
        import traceback

        try:
            if st.session_state['generated']:
                with st.container():  # Changed response_container to st.container()
                    for i in range(len(st.session_state['generated'])):
                        st.message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')

                        if len(st.session_state["problem_statement_list"][i]) < 1:
                            st.markdown(
                                """<span style="word-wrap:break-word;">No similar problem statement found in the system. Do you want to submit a new problem statement?</span>""",
                                unsafe_allow_html=True)
                            st.markdown(
                                """<span style="word-wrap:break-word;"><a href="mailto:chisel@chi.sg" target="_blank">Mail to chisel@chi.sg to submit a new problem statement</a></span>""",
                                unsafe_allow_html=True)
                        elif len(st.session_state["problem_statement_list"][i]) == 1:
                            for problem_statement_data in st.session_state["problem_statement_list"][i]:
                                score = problem_statement_data["score"]
                                year = problem_statement_data["year"]
                                category = problem_statement_data["category"]
                                requestor = problem_statement_data["requestor"].strip().replace('\n', '<br>')
                                problem_statement = problem_statement_data["problem_statement"]
                                contributor = problem_statement_data["contributor"]
                                background = problem_statement_data["background"]
                                desired_outcomes = problem_statement_data["desired_outcomes"]
                                funding = problem_statement_data["funding"]

                                st.markdown(
                                    f"""<span style="word-wrap:break-word;"><strong>Problem Statement Found:</strong> {problem_statement}</span> <span style="word-wrap:break-word; font-style: italic;">(Relevance Score: {score}%)</span>""",
                                    unsafe_allow_html=True)
                                st.markdown(f"""<span style="word-wrap:break-word;"><strong>Year:</strong> {year}</span>""",
                                            unsafe_allow_html=True)
                                st.markdown(
                                    f"""<span style="word-wrap:break-word;"><strong>Requestor/Dept/Institution:</strong><br>{requestor}</span>""",
                                    unsafe_allow_html=True)
                                st.markdown(
                                    f"""<span style="word-wrap:break-word;"><strong>Contributor:</strong><br>{contributor}</span>""",
                                    unsafe_allow_html=True)

                                with st.expander("See more"):
                                    st.markdown(f"""<span style="word-wrap:break-word;"><strong>Category:</strong> {category}</span>""",
                                                unsafe_allow_html=True)
                                    st.markdown(f"""<span style="word-wrap:break-word;"><strong>Background:</strong> {background}</span>""",
                                                unsafe_allow_html=True)
                                    st.markdown(
                                        f"""<span style="word-wrap:break-word;"><strong>Desired Outcomes:</strong> {desired_outcomes}</span>""",
                                        unsafe_allow_html=True)
                                    st.markdown(f"""<span style="word-wrap:break-word;"><strong>Funding:</strong> {funding}</span>""",
                                                unsafe_allow_html=True)

                                st.markdown(f"""<br>""", unsafe_allow_html=True)

                        else:
                            for counter, problem_statement_data in enumerate(st.session_state["problem_statement_list"][i], 1):
                                score = problem_statement_data["score"]
                                year = problem_statement_data["year"]
                                requestor = problem_statement_data["requestor"].strip().replace('\n', '<br>')
                                problem_statement = problem_statement_data["problem_statement"]
                                contributor = problem_statement_data["contributor"]
                                background = problem_statement_data["background"]
                                desired_outcomes = problem_statement_data["desired_outcomes"]
                                funding = problem_statement_data["funding"] 

                                st.markdown(
                                    f"""<span style="word-wrap:break-word;"><strong>Problem Statement Found {counter}:</strong> {problem_statement}</span> <span style="word-wrap:break-word; font-style: italic;">(Relevance Score: {score}%)</span>""",
                                    unsafe_allow_html=True)
                                st.markdown(f"""<span style="word-wrap:break-word;"><strong>Year:</strong> {year}</span>""",
                                            unsafe_allow_html=True)
                                st.markdown(
                                    f"""<span style="word-wrap:break-word;"><strong>Requestor/Dept/Institution:</strong><br>{requestor}</span>""",
                                    unsafe_allow_html=True)
                                st.markdown(
                                    f"""<span style="word-wrap:break-word;"><strong>Contributor:</strong><br>{contributor}</span>""",
                                    unsafe_allow_html=True)

                                with st.expander("See more"):
                                    st.markdown(f"""<span style="word-wrap:break-word;"><strong>Background:</strong> {background}</span>""",
                                                unsafe_allow_html=True)
                                    st.markdown(
                                        f"""<span style="word-wrap:break-word;"><strong>Desired Outcomes:</strong> {desired_outcomes}</span>""",
                                        unsafe_allow_html=True)
                                    st.markdown(f"""<span style="word-wrap:break-word;"><strong>Funding:</strong> {funding}</span>""",
                                                unsafe_allow_html=True)

                                st.markdown(f"""<br>""", unsafe_allow_html=True)

                        st.message(f'Similar projects found in CHILD: {st.session_state["child_response"][i]}', key=str(i))

                        accordion_html_code = st.session_state["accordion_html_code"][i]
                        accordion_height = st.session_state["accordion_height"][i]

                        if accordion_height > 0:
                            st.components.v1.html(accordion_html_code, height=accordion_height)
            pass
        except Exception as e:
            st.error('An error has occurred. Please try again.', icon="")
            if hasattr(e, 'message'):
                error_message = e.message
            else:
                error_message = str(e)
                
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            st.error(f'Error Type: {exc_type}', icon="")
            st.error(f'File Name: {fname}', icon="")
            st.error(f'Line Number: {exc_tb.tb_lineno}', icon="")
            
            print(traceback.format_exc())
