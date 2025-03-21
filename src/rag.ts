import { ChatOpenAI } from "@langchain/openai";
import { OpenAIEmbeddings } from '@langchain/openai';
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { z } from 'zod';

const run = async () => {

  console.log('Run the Langchain project start...');

  const searchSchema = z.object({
    query: z.string().describe("Search query to run."),
    section: z.enum(["beginning", "middle", "end"]).describe("Section to query."),
  });

  const llm = new ChatOpenAI({
    model: 'gpt-4o-mini',
    temperature: 0,
  });

  const structuredLlm = llm.withStructuredOutput(searchSchema);

  const embeddings = new OpenAIEmbeddings({
    model: 'text-embedding-3-large'
  });

  const vectorStore = new MemoryVectorStore(embeddings);

  const pTagSelector = "p";

  const cheerioLoader = new CheerioWebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/", {
    selector: pTagSelector,
  });

  const docs = await cheerioLoader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const allSplits = await splitter.splitDocuments(docs);

  const totalDocuments = allSplits.length;
  const third = Math.floor(totalDocuments / 3);

  allSplits.forEach((document, i) => {
    if (i < third) {
      document.metadata['section'] = 'beginning';
    } else if (i < third * 2) {
      document.metadata['section'] = 'middle';
    } else {
      document.metadata['section'] = 'end';
    }
  });

  await vectorStore.addDocuments(allSplits);

  const promptTemplate = await pull<ChatPromptTemplate>('rlm/rag-prompt');

  const InputStateAnnotation = Annotation.Root({
    question: Annotation<string>,
  });

  const StateAnnotation = Annotation.Root({
    question: Annotation<string>,
    search: Annotation<z.infer<typeof searchSchema>>,
    context: Annotation<Document[]>,
    answer: Annotation<string>,
  });

  const analyzeQuery = async (state: typeof InputStateAnnotation.State) => {
    const result = await structuredLlm.invoke(state.question);
    return { search: result };
  }

  const retrieve = async (
    state: typeof StateAnnotation.State,
  ) => {
    const filter = (doc: Document) => doc.metadata.section === state.search.section; 
    const retrieveDocs = await vectorStore.similaritySearch(state.search.query, 2, filter);
    return { context: retrieveDocs };
  }

  const generate = async (
    state: typeof StateAnnotation.State,
  ) => {
    const docsContent = state.context.map(doc => doc.pageContent).join('\n');
    
    const messages = await promptTemplate.invoke({
      question: state.question,
      context: docsContent,
    });
    
    const response = await llm.invoke(messages);

    return { answer: response.content };
  }

  const graph = new StateGraph(StateAnnotation)
    .addNode('analyzeQuery', analyzeQuery)
    .addNode('retrieve', retrieve)
    .addNode('generate', generate)
    .addEdge('__start__', 'analyzeQuery')
    .addEdge('analyzeQuery', 'retrieve')
    .addEdge('retrieve', 'generate')
    .addEdge('generate', '__end__')
    .compile();

  const inputs = {
    question: "What does the end of the post say about Task Decomposition?"
  };

  const result = await graph.invoke(inputs);

  console.log(result.answer);

  console.log('Run the Langchain project end...');  
};

run();