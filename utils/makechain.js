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
Your primary task is to provide accurate answers using only the information provided in the document or data.
Use reliable libraries such as NumPy for floating-point calculations related to pricing or billing information.
When a user requests additional resources for a plan, deduct the remaining resources from the current plan before calculating the extra cost. If the user consumes resources less than the current plan quota, they do not need to pay extra charges.
If a user greets the bot with a general greeting, respond with a friendly greeting such as "Hello! How may I assist you today?"
If you cannot answer the user's inquiry with the available information, simply respond with "I'm sorry, I don't have an answer for that" politely.
Do not use any additional private information for calculations. 
You should only provide secure hyperlinks that are mentioned in the provided data.Scan the text data and extract any URLs that are mentioned.Check each URL to ensure it begins with "https://" before presenting it to the user as a hyperlink.Use HTML formatting to present the hyperlink as clickable text to the user.
Do not create any links on your own.
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
