
import { ChatOpenAI } from "@langchain/openai";
import dotenv from "dotenv";
import {
  START,
  END,
  MessagesAnnotation,
  StateGraph,
  MemorySaver,
  Annotation,
} from '@langchain/langgraph';
import { v4 as uuidv4 } from 'uuid';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { trimMessages } from '@langchain/core/messages';

dotenv.config();

const chatbot = async () => {
  console.log('Run the Langchain project start...');

  const llm = new ChatOpenAI({
    model: 'gpt-4o-mini',
    temperature: 0,
  });

  const config = {
    configurable: {
      thread_id: uuidv4(),
    },
  };

  const trimmer = trimMessages({
    maxTokens: 10,
    strategy: "last",
    tokenCounter: (msgs) => msgs.length,
    includeSystem: true,
    allowPartial: false,
    startOn: "human",
  });

  const graphAnnotation = Annotation.Root({
    ...MessagesAnnotation.spec,
    language: Annotation<string>(),
  });
  
  const promptTemplate = ChatPromptTemplate.fromMessages([
    ['system', 'You are a helpful assistant. Answer all questions to the best of your ability in {language}.'],
    ['placeholder', '{messages}']
  ]);

  const callModel = async (
    state: typeof graphAnnotation.State,
  ) => {
    const messages = await trimmer.invoke(state.messages);
    const prompt = await promptTemplate.invoke({
      message: messages,
      language: state.language,
    });
    const response = await llm.invoke(prompt);

    return { messages: [response] }; 
  }

  const workflow = new StateGraph(graphAnnotation)
    .addNode('model', callModel)
    .addEdge(START, 'model')
    .addEdge('model', END);

  const app = workflow.compile({ checkpointer:  new MemorySaver() });

  const input = {
    messages: {
      role: 'user',
      content: 'Hi! I am Jim',
    },
    language: 'Spanish'
  };

  const output = await app.invoke(input, config);

  console.log(output.messages[output.messages.length -1]);

  const input2 = [
    {
      role: 'user',
      content: 'What is my name?',
    }
  ];

  const output2 = await app.invoke({ messages: input2 }, config);

  console.log(output2.messages[output2.messages.length -1]);

  console.log('Run the Langchain project end...');  
};

chatbot();