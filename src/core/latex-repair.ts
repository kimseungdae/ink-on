export function repairLatex(tokens: string[]): string[] {
  let result = [...tokens];
  result = balanceBraces(result);
  result = fixFracArgs(result);
  result = fixSqrtArgs(result);
  return result;
}

function balanceBraces(tokens: string[]): string[] {
  const result: string[] = [];
  let depth = 0;

  for (const t of tokens) {
    if (t === "{") {
      depth++;
      result.push(t);
    } else if (t === "}") {
      if (depth > 0) {
        depth--;
        result.push(t);
      }
      // drop unmatched '}'
    } else {
      result.push(t);
    }
  }

  // close unclosed '{'
  while (depth > 0) {
    result.push("}");
    depth--;
  }

  return result;
}

function fixFracArgs(tokens: string[]): string[] {
  const result: string[] = [];
  let i = 0;

  while (i < tokens.length) {
    if (tokens[i] === "\\frac") {
      result.push(tokens[i]!);
      i++;
      // collect up to 2 braced groups
      let groups = 0;
      while (i < tokens.length && groups < 2) {
        if (tokens[i] === "{") {
          groups++;
          let depth = 1;
          result.push(tokens[i]!);
          i++;
          while (i < tokens.length && depth > 0) {
            if (tokens[i] === "{") depth++;
            else if (tokens[i] === "}") depth--;
            result.push(tokens[i]!);
            i++;
          }
        } else {
          // single token arg (no braces) — only allowed as first or second arg
          result.push(tokens[i]!);
          i++;
          groups++;
        }
      }
      // skip any extra braced groups
      while (i < tokens.length && tokens[i] === "{") {
        let depth = 1;
        i++; // skip '{'
        while (i < tokens.length && depth > 0) {
          if (tokens[i] === "{") depth++;
          else if (tokens[i] === "}") depth--;
          i++;
        }
      }
    } else {
      result.push(tokens[i]!);
      i++;
    }
  }

  return result;
}

function fixSqrtArgs(tokens: string[]): string[] {
  const result: string[] = [];
  let i = 0;

  while (i < tokens.length) {
    if (tokens[i] === "\\sqrt") {
      result.push(tokens[i]!);
      i++;
      // collect exactly 1 braced group
      let groups = 0;
      while (i < tokens.length && groups < 1) {
        if (tokens[i] === "{") {
          groups++;
          let depth = 1;
          result.push(tokens[i]!);
          i++;
          while (i < tokens.length && depth > 0) {
            if (tokens[i] === "{") depth++;
            else if (tokens[i] === "}") depth--;
            result.push(tokens[i]!);
            i++;
          }
        } else {
          result.push(tokens[i]!);
          i++;
          groups++;
        }
      }
      // skip extra braced groups
      while (i < tokens.length && tokens[i] === "{") {
        let depth = 1;
        i++;
        while (i < tokens.length && depth > 0) {
          if (tokens[i] === "{") depth++;
          else if (tokens[i] === "}") depth--;
          i++;
        }
      }
    } else {
      result.push(tokens[i]!);
      i++;
    }
  }

  return result;
}

const NEEDS_ARGS = new Set([
  "\\sum",
  "\\int",
  "\\sin",
  "\\cos",
  "\\tan",
  "\\log",
  "\\lim",
  "\\sqrt",
  "\\frac",
]);

export function isCompleteExpression(tokens: string[]): boolean {
  if (tokens.length === 0) return false;
  if (tokens.length === 1 && NEEDS_ARGS.has(tokens[0]!)) return false;

  // \frac without 2 args or \sqrt without 1 arg = incomplete
  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i] === "\\frac") {
      const remaining = tokens.slice(i + 1);
      if (countBracedGroups(remaining) < 2) return false;
    }
    if (tokens[i] === "\\sqrt") {
      const remaining = tokens.slice(i + 1);
      if (countBracedGroups(remaining) < 1) return false;
    }
  }

  return true;
}

function countBracedGroups(tokens: string[]): number {
  let groups = 0;
  let i = 0;
  while (i < tokens.length) {
    if (tokens[i] === "{") {
      groups++;
      let depth = 1;
      i++;
      while (i < tokens.length && depth > 0) {
        if (tokens[i] === "{") depth++;
        else if (tokens[i] === "}") depth--;
        i++;
      }
    } else {
      break;
    }
  }
  return groups;
}
