import { OpenAI } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { PineconeStore } from 'langchain/vectorstores';
import { PromptTemplate } from 'langchain/prompts';
import { CallbackManager } from 'langchain/callbacks';

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`);

const QA_PROMPT = PromptTemplate.fromTemplate(
  `Welcome to Offer18's AI assistant! Our AI assistant provides helpful advice and custom calculations based on user queries, and is designed to be user-friendly and easy to interact with. We take security very seriously, and our AI assistant only provides secure hyperlinks that reference the context below. This helps to ensure that our users are only accessing trusted and verified information.

  Please note the following instructions:
  
  1. If you require a custom use of a source or calculation, our AI assistant will provide accurate results based on your query.
  
  2. If you have a custom demand for a plan, please note that our AI assistant is limited to the plans and information available in the context provided. If we cannot provide a satisfactory answer, we may need to refer you to our sales team. 
  
  3. Our AI assistant will not refer you to our sales team unless we cannot provide an answer from the context provided. 
  
  4. Please note that our AI assistant will not suggest any information unless it is directly related to your query. We want to ensure that the information we provide is accurate and relevant to your needs.
  
  5. If you need to perform a calculation, please provide us with the necessary information and we'll do our best to provide an accurate result. For example, if you need to calculate the cost of 30,000 units and know that the cost of an additional 10,000 units is $2, our AI assistant can use the following equation to determine the cost:
  
      Cost of 30,000 units = (Cost of 10,000 units) x 3 = ($2) x 3 = $6
  
  If you have a question, just ask us and we'll do our best to provide a helpful answer. However, please note that our AI assistant is tuned to only answer questions that are related to the context provided. If your question is outside the scope of our expertise, we'll do our best to direct you to someone who can help.
  
  

Question: {question}
=========
{context}
=========
Answer in Markdown:`,
);

export const makeChain = (
  vectorstore: PineconeStore,
  onTokenStream?: (token: string) => void,
) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAI({ temperature: 0 }),
    prompt: CONDENSE_PROMPT,
  });
  const docChain = loadQAChain(
    new OpenAI({
      temperature: 0,
      modelName: 'text-davinci-003',
      streaming: Boolean(onTokenStream),
      callbackManager: onTokenStream
        ? CallbackManager.fromHandlers({
          async handleLLMNewToken(token) {
            onTokenStream(token);
            console.log(token);
          },
        })
        : undefined,
    }), 
    { prompt: QA_PROMPT },
  );

  return new ChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
    returnSourceDocuments: true,
    k: 2, //number of source documents to return
  });
};
