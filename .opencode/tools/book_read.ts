import { tool } from "@opencode-ai/plugin"
import path from "path"

export default tool({
  description:
    "Read exact RU canonical text for one logical/chunk ID with provenance, checksum and version.",
  args: {
    chunk_id: tool.schema.string().describe("Logical or physical RU chunk ID"),
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
      expected_ru_version: args.expected_ru_version ?? null,
    })
    const result =
      await Bun.$`python3 ${script} book_read --index-dir ${indexDir} --input-json ${input}`.text()
    return result.trim()
  },
})
