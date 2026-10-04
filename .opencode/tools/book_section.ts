import { tool } from "@opencode-ai/plugin"
import path from "path"

export default tool({
  description:
    "Bounded paginated larger RU section read of whole exact chunks under the source-token ceiling.",
  args: {
    section_id: tool.schema.string().describe("Language-neutral section id"),
    chunk_offset: tool.schema.number().describe("Chunk offset for pagination"),
    chunk_limit: tool.schema.number().describe("Chunks to return (1..12)"),
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
      section_id: args.section_id,
      chunk_offset: args.chunk_offset,
      chunk_limit: args.chunk_limit,
      expected_ru_version: args.expected_ru_version ?? null,
    })
    const result =
      await Bun.$`python3 ${script} book_section --index-dir ${indexDir} --input-json ${input}`.text()
    return result.trim()
  },
})
