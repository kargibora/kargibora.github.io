import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import tailwind from "@astrojs/tailwind";
import remarkMath from 'remark-math';
import rehypeMathJax from 'rehype-mathjax';
import remarkGfm from 'remark-gfm';
import remarkSmartypants from 'remark-smartypants';
import rehypeKatex from 'rehype-katex';
// https://astro.build/config
export default defineConfig({
  site: 'https://kargibora.github.io',
  integrations: [mdx(), sitemap(), tailwind()],
  markdown: {
    //mode: 'mdx',
    remarkPlugins: [
      remarkGfm, remarkSmartypants, remarkMath
    ],
    rehypePlugins: [
      //'rehype-slug', < needed only prior beta.22
       rehypeKatex, 
    ]
  }
});