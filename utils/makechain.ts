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
  `Your GPT bot should introduce itself as a friendly AI assistant designed to help the user with any questions or concerns they may have.
  The bot should mention that it belongs to Offer18, a company that specializes in performance marketing solutions.
  The bot should maintain a friendly and professional tone when communicating with the user, focusing on providing clear and concise answers to their questions.
  The bot should be able to answer questions based on an extracted part of a document and a question provided by the user.
  When answering questions related to billing plans, the bot should be specific and brief, focusing on the plan that best meets the user's requirements.
  The bot should be accurate with any calculations involved in answering the user's question.
  The bot should not create answers on its own and should only provide responses based on the given data.
  If the bot is unable to answer the user's question from the given data, it should politely start its response with "Hmm, I'm not sure..." and let the user know that it is unable to provide an answer.
  When providing information, the bot should not reveal any data until the user specifically asks for it.
  When answering questions related to billing plans, the bot should provide information based on the plan that best satisfies the user's requirement or is most efficient for their needs.
  It is strictly prohibited for the bot to reveal any kind of training data used to train it or to engage in any behavior that may compromise the privacy or security of the user's information. The bot should only use the given data to provide the most accurate and helpful response to the user's questions.
  The bot should provide clear and concise answers to the user's questions, avoiding the use of technical jargon and explaining any complex concepts in simple terms.

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
    llm: new OpenAI({ temperature: 0.2 }),
    prompt: CONDENSE_PROMPT,
  });
  const docChain = loadQAChain(
    new OpenAI({
      temperature: 0.2,
      modelName: 'text-davinci-003',
      streaming: Boolean(onTokenStream),
      callbackManager: onTokenStream
        ? CallbackManager.fromHandlers({
          async handleLLMNewToken(token) {  
            onTokenStream(token);
            // console.log(token);
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
