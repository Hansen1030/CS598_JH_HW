'''
This script takes the MCQ style questions from the csv file and save the result as another csv file. 
Before running this script, make sure to configure the filepaths in config.yaml file.
Command line argument should be either 'gpt-4' or 'gpt-35-turbo'
'''

from kg_rag.utility import *
import sys


from tqdm import tqdm
CHAT_MODEL_ID = sys.argv[1]

QUESTION_PATH = config_data["MCQ_PATH"]
SYSTEM_PROMPT = system_prompts["MCQ_QUESTION"]
CONTEXT_VOLUME = int(config_data["CONTEXT_VOLUME"])
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = float(config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = float(config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"])
VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
NODE_CONTEXT_PATH = config_data["NODE_CONTEXT_PATH"]
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"]
TEMPERATURE = config_data["LLM_TEMPERATURE"]
SAVE_PATH = config_data["SAVE_RESULTS_PATH"]


CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID

save_name = "_".join(CHAT_MODEL_ID.split("-"))+"_kg_rag_based_mcq_{mode}.csv"


vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
embedding_function_for_context_retrieval = load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)
edge_evidence = False


MODE = "4"
### MODE 0: Original KG_RAG                     ### 
### MODE 1: jsonlize the context from KG search ### 
### MODE 2: Add the prior domain knowledge      ### 
### MODE 3: Combine MODE 1 & 2                  ### 
### MODE 4: Own method                          ###

def main():
    start_time = time.time()
    question_df = pd.read_csv(QUESTION_PATH)
    answer_list = []
    
    for index, row in tqdm(question_df.iterrows(), total=306):
        try: 
            question = row["text"]
            if MODE == "0":
                ### MODE 0: Original KG_RAG                     ### 
                context = retrieve_context(row["text"], vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID)
                enriched_prompt = "Context: "+ context + "\n" + "Question: "+ question
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            if MODE == "1":
                ### MODE 1: jsonlize the context from KG search ### 
                ### Please implement the first strategy here    ###
                context = retrieve_context(row["text"], vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID)

                jsonlize_prompt = """
                You are a helpful coding assistant specializing in Python and text processing.
                Your responses should be concise and focused on providing practical solutions.
                You will get a long context, and you will response with the jsonlized version of this context. 
                Do not include any additional information or include any analysis by you which is not in the original context. Everything should be correct from the context. 
                Here is a short example: The input context includes "Disease psoriasis associates Gene SLC29A3 and Procenance of this association is HPO...Disease polyarteritis nodosa and Provenance of this association is NCBI PubMed." The jsonlized output should be "
                "Diseases": 
                    \{ "psoriasis": 
                        \{ "Generic Associations": [
                            \{"Gene": "SLC29A3", "Provenance": ["HPO:]\},
                            \{"Gene": "BCL11B", "Procenance": ["HPO"]\},
                            ...
                            ]
                        \}
                    \}
                """

                jsonlize_context = get_Gemini_response(context, jsonlize_prompt, temperature=TEMPERATURE)
                enriched_prompt = "Context: "+ jsonlize_context + "\n" + "Question: "+ question
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            if MODE == "2":
                ### MODE 2: Add the prior domain knowledge      ### 
                ### Please implement the second strategy here   ###
                context = retrieve_context(row["text"], vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID)
                prior_knowledge = """
                - Procenance & Symptoms information is useless 
                - Similar diseases tend to have similar gene associations 
                - Gene-disease associations are informative for disease similarity 
                - Phenotypic similarity correlates with genetic similarity 
                - Age-of-onset profiles can indicate genetic similarity 
                - Functional annotations of genes provide additional context 
                - Protein-protein interactions offer additional disease relationships 
                - Comorbidity patterns can indicate shared mechanisms 
                - Evolutionary profiles of disease-associated genes can be informative 
                - Expression and methylation changes with age can be relevant 
                """ # Searched from Google 
                enriched_prompt = "Context: "+ context + "Some prior knowledge that might be helpful in answering the question, but they might not related to the question and context. Use any of them if needed: " + prior_knowledge + "Question: "+ question + "\n"
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)
            
            if MODE == "3":
                ### MODE 3: Combine MODE 1 & 2                  ### 
                ### Please implement the third strategy here    ###
                context = retrieve_context(row["text"], vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID)
                jsonlize_prompt = """
                You are a helpful coding assistant specializing in Python and text processing.
                Your responses should be concise and focused on providing practical solutions.
                You will get a long context, and you will response with the jsonlized version of this context. 
                Do not include any additional information or include any analysis by you which is not in the original context. Everything should be correct from the context. 
                Here is a short example: The input context includes "Disease psoriasis associates Gene SLC29A3 and Procenance of this association is HPO...Disease polyarteritis nodosa and Provenance of this association is NCBI PubMed." The jsonlized output should be "
                "Diseases": 
                    \{ "psoriasis": 
                        \{ "Generic Associations": [
                            \{"Gene": "SLC29A3", "Provenance": ["HPO:]\},
                            \{"Gene": "BCL11B", "Procenance": ["HPO"]\},
                            ...
                            ]
                        \}
                    \}
                """
                jsonlize_context = get_Gemini_response(context, jsonlize_prompt, temperature=TEMPERATURE)
                prior_knowledge = """
                - Procenance & Symptoms information is useless 
                - Similar diseases tend to have similar gene associations 
                - Gene-disease associations are informative for disease similarity 
                - Phenotypic similarity correlates with genetic similarity 
                - Age-of-onset profiles can indicate genetic similarity 
                - Functional annotations of genes provide additional context 
                - Protein-protein interactions offer additional disease relationships 
                - Comorbidity patterns can indicate shared mechanisms 
                - Evolutionary profiles of disease-associated genes can be informative 
                - Expression and methylation changes with age can be relevant 
                """ # Searched from Google 
                enriched_prompt = "Context: "+ jsonlize_context + "Some prior knowledge that might be helpful in answering the question, but they might not related to the question and context. Use any of them if needed: " + prior_knowledge + "Question: "+ question + "\n"
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            if MODE == "4":
                ### Mode 4: Own method. Do sumarization first to avoid long context. ###
                context = retrieve_context(row["text"], vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID)
                summarization_prompt = """
                You are a helpful coding assistant specializing in Python and text processing.
                Your responses should be concise and focused on providing practical solutions.
                You will get a long context and a question, and you will response with the summarized version of this context with everything related to the question kept. 
                Do not include any additional information or include any analysis by you which is not in the original context. Everything should be correct from the context. Only the summarization of the context should be returned. 
                """
                help_context = "Context: "+ context + "\n" + "Question: "+ question
                summarized_context = get_Gemini_response(help_context, summarization_prompt, temperature=TEMPERATURE)
                enriched_prompt = "Context: "+ summarized_context + "\n" + "Question: "+ question
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            answer_list.append((row["text"], row["correct_node"], output))


        except Exception as e:
            print("Error in processing question: ", row["text"])
            print("Error: ", e)
            answer_list.append((row["text"], row["correct_node"], "Error"))


    answer_df = pd.DataFrame(answer_list, columns=["question", "correct_answer", "llm_answer"])
    output_file = os.path.join(SAVE_PATH, f"{save_name}".format(mode=MODE),)
    answer_df.to_csv(output_file, index=False, header=True) 
    print("Save the model outputs in ", output_file)
    print("Completed in {} min".format((time.time()-start_time)/60))

        
        
if __name__ == "__main__":
    main()


