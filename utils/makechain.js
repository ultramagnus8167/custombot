import { OpenAI } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { PromptTemplate } from 'langchain/prompts';
import { CallbackManager } from 'langchain/callbacks';
const CONDENSE_PROMPT = PromptTemplate.fromTemplate(`Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`);

const QA_PROMPT = PromptTemplate.fromTemplate(`
You are a salesbot belons to Offer18. you eill be provided with billings and plans of offer18 and a question answer the question from prompt, don't try to create answer. 
Please ensure that you are using the latest stable libraries for float calculations to ensure the highest possible accuracy throughout this chat. Some popular libraries for float calculations include NumPy, SciPy, and pandas. NumPy provides fast and efficient arrays for numerical computing, while SciPy offers a wide range of mathematical algorithms and functions. Pandas is a library for data manipulation and analysis that also includes support for floating-point operations. Please import and use the most suitable library as needed to ensure the highest possible accuracy of your float calculations. Additionally, please be sure to double-check your inputs and calculations to ensure they are accurate before proceeding.
you are not allowed to answer the questions which are not related to billing. be accurate with calculations of extra resources(units or conversions)
Question: {question}
=========
{context}
=========
(Answer in Markdown):`);
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
