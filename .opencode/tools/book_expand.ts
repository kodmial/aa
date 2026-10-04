import { tool } from "@opencode-ai/plugin"
import path from "path"

export default tool({
  description:
    "Bounded neighboring exact RU chunks around one chunk ID. Hard max is 3 before + 3 after.",
  args: {
    chunk_id: tool.schema.string().describe("Logical or physical RU chunk ID"),
    before: tool.schema.number().describe("Chunks before (0..3)"),
    after: tool.schema.number().describe("Chunks after (0..3)"),
    expected_ru_version: tool.schema
      .string()
      .optional()
      .describe("Optional pinned RU artifact SHA; mismatch fails closed"),
  },
  async execute(args, context) {
    const root = context.worktree || context.directory
    const script = path.join(root, "scripts", "aa_book_tool.py")
    const indexDir = path.join(root, "corpus", "generated", "retrieval")
    const input = JSON.stringify({
      chunk_id: args.chunk_id,
      before: args.before,
      after: args.after,
      expected_ru_version: args.expected_ru_version ?? null,
    })
    const result =
      await Bun.$`python3 ${script} book_expand --index-dir ${indexDir} --input-json ${input}`.text()
    return result.trim()
  },
})
