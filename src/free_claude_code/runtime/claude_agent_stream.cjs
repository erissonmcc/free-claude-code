const { randomUUID } = require('node:crypto');

// SDK assistant frames contain one completed block. Partial frames identify it
// by message ID + block index and arrive before that assistant frame.
class ClaudeTextStream {
  constructor() {
    this.messageId = null;
    this.blocks = new Map();
    this.assistantBlockIndex = 0;
  }

  frame(block) {
    return { type: 'text', id: block.id, text: block.text, complete: block.complete };
  }

  accept(message) {
    const output = [];
    if (message.type === 'stream_event' && !message.parent_tool_use_id) {
      const event = message.event;
      if (event.type === 'message_start') {
        this.messageId = event.message.id;
        this.blocks.clear();
        this.assistantBlockIndex = 0;
      } else if (event.type === 'content_block_start') {
        if (event.content_block.type === 'text') {
          this.blocks.set(event.index, {
            id: `text:${this.messageId}:${event.index}`,
            text: event.content_block.text || '',
            complete: false,
          });
        }
      } else if (event.type === 'content_block_delta' && event.delta.type === 'text_delta') {
        const block = this.blocks.get(event.index);
        if (block && !block.complete) {
          block.text += event.delta.text;
          output.push(this.frame(block));
        }
      } else if (event.type === 'content_block_stop' || event.type === 'message_stop') {
        const blocks = event.type === 'message_stop' ? [...this.blocks.values()] : [this.blocks.get(event.index)];
        for (const block of blocks) {
          if (block && !block.complete && block.text) {
            block.complete = true;
            output.push(this.frame(block));
          }
        }
      }
    } else if (message.type === 'assistant') {
      const topLevel = !message.parent_tool_use_id && message.message.id === this.messageId;
      // SDK confirmations carry completed blocks, not the full message content.
      // Count thinking/tool blocks too so their original stream indexes survive.
      for (const [offset, content] of (message.message.content || []).entries()) {
        const index = topLevel ? this.assistantBlockIndex++ : offset;
        if (content.type !== 'text') continue;
        const streamed = topLevel ? this.blocks.get(index) : null;
        if (streamed) {
          if (streamed.complete && streamed.text === content.text) continue;
          streamed.text = content.text;
          streamed.complete = true;
          output.push(this.frame(streamed));
        } else {
          output.push({ type: 'text', id: `text:${message.uuid || randomUUID()}:${index}`, text: content.text, complete: true });
        }
      }
    } else if (message.type === 'result') {
      for (const block of this.blocks.values()) {
        if (!block.complete && block.text) {
          block.complete = true;
          output.push(this.frame(block));
        }
      }
    }
    return output;
  }
}

class ClaudeUsageStream {
  constructor() {
    this.messages = new Map();
    this.current = null;
    this.last = null;
  }

  accept(message) {
    if (message.type !== 'stream_event' || message.parent_tool_use_id) return null;
    const event = message.event;
    if (event.type === 'message_start') {
      this.current = event.message.id;
      if (!this.messages.has(this.current)) this.messages.set(this.current, {});
    }
    const usage = event.type === 'message_start' ? event.message.usage :
      event.type === 'message_delta' ? event.usage : null;
    if (!usage || !this.current) return null;
    const current = this.messages.get(this.current);
    for (const key of ['input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens']) {
      if (Number.isSafeInteger(usage[key]) && usage[key] >= 0) current[key] = usage[key];
    }
    const total = [...this.messages.values()].reduce((sum, entry) =>
      sum + Object.values(entry).reduce((a, b) => a + b, 0), 0);
    if (total === this.last) return null;
    this.last = total;
    return total;
  }
}

module.exports = { ClaudeTextStream, ClaudeUsageStream };
