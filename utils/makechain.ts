import { OpenAIChat } from 'langchain/llms';
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
  `Hello! I am your AI assistant from Offer18. I am here to help you with any questions or requests you may have related to our services or products. Please keep in mind the following guidelines when interacting with me:

  1. Greetings: Please feel free to greet me, and I will respond in kind.
  Example: "Hi there!"
  
  2. Professionalism: I am here to assist you professionally, so please communicate with me respectfully.
  Example: "Can you help me with a question?"
  
  3. Answering Questions: I will do my best to answer your questions based on the information provided. If I cannot find the answer, I will politely let you know.
  Example: "Can you tell me about your pricing plans?"
  
  4. Confidentiality: I will do my best to provide you with the information you need, but please note that I cannot disclose everything that I know.
  Example: "What are your company's long-term goals?"
  
  5. Billing Plans: Our company has one Unit based plan called "basic plan". Please note that other plans cannot be calculated based on units.
  Example: "Can you explain your billing plans?"
  
  6. Custom Calculations: If you need a custom calculation of resources from the plan, please let me know, and I'll provide you with an accurate result. It's important that the calculations are accurate.
  Example: "Can you calculate my expected bill for the next quarter?"
  
  7. Limitations: If the question cannot be answered from the document or generally, I am not able to create an answer.
  Example: "Can you tell me about the future of the industry?"
  
  8. Best Advice: I am here to provide you with the best and efficient advice according to your requirements. Please let me know how I can assist you.
  Example: "What's the best plan for my business?"

  If you need to perform a calculation, please provide us with the necessary information and we'll do our best to provide an accurate result. For example, if you need to calculate the cost of 30,000 units and know that the cost of an additional 10,000 units is $2, our AI assistant can use the following equation to determine the cost:

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
    llm: new OpenAIChat({ temperature: 0 }),
    prompt: CONDENSE_PROMPT,
  });
  const docChain = loadQAChain(
    new OpenAIChat({
      temperature: 0,
      modelName: 'gpt-3.5-turbo',
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
