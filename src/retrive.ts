import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from '@langchain/openai';

const retrieve = async () => {
  console.log('Run the Langchain project...');

  const loader = new PDFLoader('./nke-10k-2023.pdf');

  const docs = await loader.load();

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const allSplits = await textSplitter.splitDocuments(docs);

  const embeddings = new OpenAIEmbeddings({
    model: 'text-embedding-3-large'
  });

  const [firstPage, secondPage] = allSplits;

  await embeddings.embedQuery(firstPage.pageContent);
  await embeddings.embedQuery(secondPage.pageContent);

  const vectorStore = new MemoryVectorStore(embeddings);

  await vectorStore.addDocuments(allSplits);

  const [incorporated] = await vectorStore.similaritySearch("When was Nike incorporated?");

  const [revenue] = await vectorStore.similaritySearchWithScore(
    "What was Nike's revenue in 2023?"
  );

  const embedding = await embeddings.embedQuery("How were Nike's margins impacted in 2023?");

  // const result = await vectorStore.similaritySearchVectorWithScore(embedding, 1);

  const retriever = await vectorStore.asRetriever({
    searchType: 'mmr',
    searchKwargs: {
      fetchK: 1,
    }
  });

  const result = await retriever.batch([
    "When was Nike incorporated?",
    "What was Nike's revenue in 2023?",
  ]);

  console.log(result);

  // console.log(result, 'result');

  console.log('Run function end...')
};