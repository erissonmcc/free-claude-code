const { randomUUID } = require('node:crypto');

// SDK assistant frames contain one completed block. Partial frames identify it
// by message ID + block index and arrive before that assistant frame.
class ClaudeTextStream {
  constructor() {
    this.messageId = null;
    this.blocks = new Map();
    this.activeIndex = null;
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
        this.activeIndex = null;
      } else if (event.type === 'content_block_start') {
        if (event.content_block.type === 'text') {
          this.activeIndex = event.index;
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
      for (const [index, content] of (message.message.content || []).entries()) {
        if (content.type !== 'text') continue;
        const streamed = !message.parent_tool_use_id && message.message.id === this.messageId
          ? this.blocks.get(this.activeIndex) : null;
        if (streamed) {
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
