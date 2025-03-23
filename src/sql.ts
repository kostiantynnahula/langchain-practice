import { SqlDatabase } from 'langchain/sql_db';
import { DataSource } from 'typeorm';
import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { pull } from 'langchain/hub';
import { z } from 'zod';
import { Annotation, MemorySaver, StateGraph } from '@langchain/langgraph';
import { QuerySqlTool } from "langchain/tools/sql";
import { SqlToolkit } from "langchain/agents/toolkits/sql";

import dotenv from "dotenv";
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { AIMessage, BaseMessage, isAIMessage } from "@langchain/core/messages";

const prettyPrint = (message: BaseMessage) => {
  let txt = `[${message._getType()}]: ${message.content}`;
  if ((isAIMessage(message) && message.tool_calls?.length) || 0 > 0) {
    const tool_calls = (message as AIMessage)?.tool_calls
      ?.map((tc) => `- ${tc.name}(${JSON.stringify(tc.args)})`)
      .join("\n");
    txt += ` \nTools: \n${tool_calls}`;
  }
  console.log(txt);
};


dotenv.config();

const sql = async () => {

  console.log('Run the Langchain project start...');

  const datasource = new DataSource({
    type: "sqlite",
    database: "Chinook.db",
  });

  const db = await SqlDatabase.fromDataSourceParams({
    appDataSource: datasource,
  });

  const artists = await db.run("SELECT * FROM Artist LIMIT 10;");

  const llm = new ChatOpenAI({
    model: 'gpt-4o-mini',
    temperature: 0,
  });

  const queryPromptTemplate = await pull<ChatPromptTemplate>('langchain-ai/sql-query-system-prompt');
  
  const queryOutput = z.object({
    query: z.string().describe("Syntactically valid SQL query."),
  });
  
  const InputStateAnnotation = Annotation.Root({
    question: Annotation<string>,
  });
  
  const StateAnnotation = Annotation.Root({
    question: Annotation<string>,
    query: Annotation<string>,
    result: Annotation<string>,
    answer: Annotation<string>,
  });

  const structuredLlm = llm.withStructuredOutput(queryOutput);

  const writeQuery = async (state: typeof InputStateAnnotation.State) => {
    const promptValue = await queryPromptTemplate.invoke({
      dialect: db.appDataSourceOptions.type,
      top_k: 10,
      table_info: await db.getTableInfo(),
      input: state.question,
    });

    const result = await structuredLlm.invoke(promptValue);
    return { query: result.query };
  }

  const employeeQuery = await writeQuery({ question: "How many Employees are there?" });

  // console.log(employeeQuery);

  const executeQuery = async (state: typeof StateAnnotation.State) => {
    const executeQueryTool = new QuerySqlTool(db);
    return {
      result: await executeQueryTool.invoke(state.query),
    };
  }

  const generateAnswer = async (state: typeof StateAnnotation.State) => {
    const promptValue =
      "Given the following user question, corresponding SQL query, " +
      "and SQL result, answer the user question.\n\n" +
      `Question: ${state.question}\n` +
      `SQL Query: ${state.query}\n` +
      `SQL Result: ${state.result}\n`;
    const response = await llm.invoke(promptValue);
    return { answer: response.content };
  };

  // const graphBuilder = new StateGraph({
  //   stateSchema: StateAnnotation,
  // })
  //   .addNode("writeQuery", writeQuery)
  //   .addNode("executeQuery", executeQuery)
  //   .addNode("generateAnswer", generateAnswer)
  //   .addEdge("__start__", "writeQuery")
  //   .addEdge("writeQuery", "executeQuery")
  //   .addEdge("executeQuery", "generateAnswer")
  //   .addEdge("generateAnswer", "__end__");

  // const graph = graphBuilder.compile();

  // let inputs = { question: "How many employees are there?" };

  // console.log(inputs);
  // console.log("\n====\n");

  // for await (const step of await graph.stream(inputs, {
  //   streamMode: "updates",
  // })) {
  //   console.log(step);
  //   console.log("\n====\n");
  // }

  // const checkpointer = new MemorySaver();

  // const graphWithInterrupt = graphBuilder.compile({
  //   checkpointer: checkpointer,
  //   interruptBefore: ['executeQuery'],
  // });

  // const threadConfig = {
  //   configurable: { thread_id: "1" },
  //   streamMode: "updates" as const,
  // };


  const toolkit = new SqlToolkit(db, llm);

  const tools = toolkit.getTools();

  console.log(
    tools.map((tool) => ({
      name: tool.name,
      description: tool.description,
    }))
  );

  const systemPromptTemplate = await pull<ChatPromptTemplate>('langchain-ai/sql-agent-system-prompt');

  const systemMessage = await systemPromptTemplate.format({
    dialect: "SQLite",
    top_k: 5,
  });

  const agent = createReactAgent({
    llm,
    tools,
    stateModifier: systemMessage
  });

  let inputs2 = {
    messages: [
      { role: "user", content: "Which country's customers spent the most?" },
    ],
  };
  
  for await (const step of await agent.stream(inputs2, {
    streamMode: "values",
  })) {
    const lastMessage = step.messages[step.messages.length - 1];
    prettyPrint(lastMessage);
    console.log("-----\n");
  }


  let inputs3 = {
    messages: [{ role: "user", content: "Describe the playlisttrack table" }],
  };
  
  for await (const step of await agent.stream(inputs3, {
    streamMode: "values",
  })) {
    const lastMessage = step.messages[step.messages.length - 1];
    prettyPrint(lastMessage);
    console.log("-----\n");
  }

  console.log('Run the Langchain project end...');  
};

sql();