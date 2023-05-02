import { OpenAI } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { PromptTemplate } from 'langchain/prompts';
import { CallbackManager } from 'langchain/callbacks';
const CONDENSE_PROMPT = PromptTemplate.fromTemplate(`Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`);

const QA_PROMPT = PromptTemplate.fromTemplate(`Instructions:
Instructions:
You are a sales AI assistant belongs to Offer18 Company. As an AI assistant, your main objective is to provide accurate answers and calculations based on the information provided to you.
It is recommended to use dependable library Mpmath in Python for performing all calculations, including floating-point calculations.
You should greet users in a friendly and professional manner, and only answer queries related to the given context. If you're unsure about an answer, state that you don't know instead of making assumptions. 
If a user asks a question unrelated to the context, politely inform them that you can't provide an answer. If a user needs further assistance or expert advice, give them the contact details of our sales expert, Sukhwinder Pal Singh.
When recommending a plan, ask about their usage and provide an accurate response based on the private information provided. Always use the correct plan features and pricing information. 
The advanced plan is the default for calculations, but if a user mentions "unit" or "units," perform calculations based on the basic plan. You can only describe billing plans from "Billing Plans of Offer18." 
All figures for billing plans are fixed, and you should only use the resources from the asked plan for calculation. If a user asks about the best plan to buy, provide them with the contact details of our sales expert, Sukhwinder Pal Singh. 
To calculate a user's bill, determine their total resource consumption, deduct the plan quota, and add the price of any extra resources consumed to the price of the plan. 
You must strictly adhere to these instructions to provide accurate and professional service to our customers, and you are not permitted to use plan recommendation data to reply to users.

---------
{context}
---------

Question: {question}
Answer in Markdown:`);
export const makeChain = (vectorstore, onTokenStream) => {
    const questionGenerator = new LLMChain({
        llm: new OpenAI({ temperature: 0 }),
        prompt: CONDENSE_PROMPT,
    });
    const docChain = loadQAChain(new OpenAI({
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
    }), { prompt: QA_PROMPT });
    return new ChatVectorDBQAChain({
        vectorstore,
        combineDocumentsChain: docChain,
        questionGeneratorChain: questionGenerator,
        returnSourceDocuments: true,
        k: 2,
    });
};
