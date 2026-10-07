import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

// GitHub-flavoured markdown (tables, strikethrough, task lists, autolinks) because LLM answers use it a lot.
// Raw HTML in model output is not rendered (react-markdown escapes it by default).
const remarkPlugins = [remarkGfm];

const components: Components = {
  a(props) {
    const { node, ...rest } = props;
    void node;
    return <a {...rest} target="_blank" rel="noreferrer noopener" />;
  },
};

export function Markdown({ text }: { text: string }) {
  return (
    <div className="md">
      <ReactMarkdown remarkPlugins={remarkPlugins} components={components}>
        {text}
      </ReactMarkdown>
    </div>
  );
}
