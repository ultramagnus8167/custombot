import { OpenAI } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { PromptTemplate } from 'langchain/prompts';
import { CallbackManager } from 'langchain/callbacks';
const CONDENSE_PROMPT = PromptTemplate.fromTemplate(`Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`);

const QA_PROMPT = PromptTemplate.fromTemplate(`Your main task is to provide answers based on the provided data, using the most reliable libraries for calculations such as the NumPy Python library, to ensure accuracy in floating-point calculations involving pricing or billing information. . 
Please provide responses based solely on the text extracted from the provided document or data. When someone greets the bot with a general greeting, such as 'Hi' or 'Hello', respond with a friendly greeting such as 'Hello! How may I assist you today?' If you're unable to respond to the user's inquiry, simply respond with 'I'm sorry, I don't have an answer for that in a polite tone.

=========
{context}
=========
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
