import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { Markdown } from "./Markdown";

describe("Markdown", () => {
  it("renders GitHub-flavoured tables, strikethrough and task lists", () => {
    const html = renderToStaticMarkup(
      <Markdown text={"| Model | Rank |\n| --- | --- |\n| alpha | 1 |\n\n~~old~~\n\n- [x] done\n- [ ] todo"} />,
    );
    expect(html).toContain("<table>");
    expect(html).toContain("<td>alpha</td>");
    expect(html).toContain("<del>old</del>");
    expect(html).toMatch(/<input[^>]*type="checkbox"[^>]*checked/);
  });

  it("escapes raw HTML from model output", () => {
    const html = renderToStaticMarkup(<Markdown text={'<img src=x onerror="alert(1)">'} />);
    expect(html).not.toContain("<img");
  });
});
