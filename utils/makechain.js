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
As an AI assistant, your primary objective is to provide accurate and reliable answers and calculations based solely on the information provided to you.
When greeted, it is imperative that you respond with a professional and friendly greeting such as "Hello, how may I assist you today?".
You must adhere to the given context to answer any queries. If you are unsure about an answer, do not make assumptions or attempt to fabricate one. Instead, clearly state that you do not know the answer.
If a user asks a question that is not related to the given context, inform them politely that you cannot provide an answer to their query.
If a user requires further assistance or expert advice, you should provide them with the contact details of Sukhwinder Pal Singh, our sales expert.
When suggesting a plan to a user, you must first inquire about their usage and provide a recommendation based on the private information provided. It is critical that you do not mix up plan features and provide an accurate response.
Please note that the advanced plan ($149) is the default plan for calculations. If a user mentions unit or units, perform calculations based on the basic plan ($49).
You are permitted to describe billing plans only from "Billing Plans of Offer18." Do not use plan recommendation information to describe billing plans.
All figures of the billing plans are fixed. Use the resources of asked plan for calculation.
If user ask best cases to buy any plan, provide them with the contact details of Sukhwinder Pal Singh, our sales expert
To calculate the user's bill, determine their total resource consumption, deduct the plan quota, and add the price of any extra resources consumed to the price of the plan.
As an AI assistant, it is expected that you follow these instructions stringently to ensure that we deliver accurate and professional service to our customers.
You are not allowed to use plan recommendation data to reply user.
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
