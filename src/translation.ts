import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';

export const translation = async (text: string, language: string) => {
  console.log('Run the Langchain project...');

  const model = new ChatOpenAI({ model: "gpt-4" });

  const systemTemplate = "Translate the following from English into {language}";

  const promptTemplate = ChatPromptTemplate.fromMessages([
    ["system", systemTemplate],
    ["user", "{text}"],
  ]);

  const promptValue = await promptTemplate.invoke({
    language,
    text,
  });

  const response = await model.invoke(promptValue);

  console.log(response)
}
