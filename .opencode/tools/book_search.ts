import { tool } from "@opencode-ai/plugin"
import path from "path"

export default tool({
  description:
    "RU-first AA retrieval: compact navigation candidates with logical IDs and RU locators. Previews are navigation only, never evidence.",
  args: {
    aspect_id: tool.schema.string().describe("Retrieval aspect id"),
    semantic_query_ru: tool.schema.string().describe("Russian semantic query"),
    lexical_queries_ru: tool.schema
      .array(tool.schema.string())
      .describe("Russian lexical queries"),
    lexical_query_en: tool.schema
      .string()
      .nullable()
      .optional()
      .describe("Must stay null: EN secondary is disabled"),
  },
  async execute(args, context) {
    const root = context.worktree || context.directory
    const script = path.join(root, "scripts", "aa_book_tool.py")
    const indexDir = path.join(root, "corpus", "generated", "retrieval")
    const input = JSON.stringify({
      aspect_id: args.aspect_id,
      semantic_query_ru: args.semantic_query_ru,
      lexical_queries_ru: args.lexical_queries_ru,
      lexical_query_en: args.lexical_query_en ?? null,
    })
    const proc = Bun.spawn(
      ["python3", script, "book_search", "--index-dir", indexDir, "--input-json", input],
      { stdout: "pipe", stderr: "pipe" },
    )
    const [stdout, stderr, exitCode] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
      proc.exited,
    ])
    if (exitCode !== 0) {
      throw new Error(stderr.trim() || `book_search exited with code ${exitCode}`)
    }
    return stdout.trim()
  },
})
