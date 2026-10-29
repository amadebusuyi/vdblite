package io.vdblite;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The JSON the native library speaks: objects become {@code Map<String, Object>},
 * arrays {@code List<Object>}, integers {@code Long}, other numbers {@code Double}.
 */
final class Json {
  private Json() {}

  // ------------------------------------------------------------ writing

  static String write(Object v) {
    StringBuilder sb = new StringBuilder();
    write(sb, v);
    return sb.toString();
  }

  private static void write(StringBuilder sb, Object v) {
    if (v == null) {
      sb.append("null");
    } else if (v instanceof String s) {
      string(sb, s);
    } else if (v instanceof Boolean b) {
      sb.append(b);
    } else if (v instanceof Double d) {
      number(sb, d);
    } else if (v instanceof Float f) {
      number(sb, f.doubleValue());
    } else if (v instanceof Number n) {
      sb.append(n.longValue());
    } else if (v instanceof Map<?, ?> m) {
      sb.append('{');
      boolean first = true;
      for (Map.Entry<?, ?> e : m.entrySet()) {
        if (!first) sb.append(',');
        first = false;
        string(sb, String.valueOf(e.getKey()));
        sb.append(':');
        write(sb, e.getValue());
      }
      sb.append('}');
    } else if (v instanceof Collection<?> c) {
      sb.append('[');
      boolean first = true;
      for (Object x : c) {
        if (!first) sb.append(',');
        first = false;
        write(sb, x);
      }
      sb.append(']');
    } else if (v instanceof Object[] a) {
      write(sb, List.of(a));
    } else {
      throw new IllegalArgumentException("can't write " + v.getClass().getName() + " as JSON");
    }
  }

  private static void number(StringBuilder sb, double d) {
    if (Double.isNaN(d) || Double.isInfinite(d)) {
      throw new IllegalArgumentException("JSON has no " + d);
    }
    if (d == Math.rint(d) && Math.abs(d) < 1e15) {
      sb.append((long) d);
    } else {
      sb.append(d);
    }
  }

  private static void string(StringBuilder sb, String s) {
    sb.append('"');
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      switch (c) {
        case '"' -> sb.append("\\\"");
        case '\\' -> sb.append("\\\\");
        case '\n' -> sb.append("\\n");
        case '\r' -> sb.append("\\r");
        case '\t' -> sb.append("\\t");
        default -> {
          if (c < 0x20) {
            sb.append(String.format("\\u%04x", (int) c));
          } else {
            sb.append(c);
          }
        }
      }
    }
    sb.append('"');
  }

  // ------------------------------------------------------------ reading

  static Object parse(String s) {
    Parser p = new Parser(s);
    Object v = p.value();
    p.ws();
    if (p.i != s.length()) throw p.error("trailing characters");
    return v;
  }

  private static final class Parser {
    final String s;
    int i;

    Parser(String s) {
      this.s = s;
    }

    IllegalStateException error(String what) {
      return new IllegalStateException("bad JSON from vdblite at " + i + ": " + what);
    }

    void ws() {
      while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++;
    }

    Object value() {
      ws();
      if (i >= s.length()) throw error("unexpected end");
      char c = s.charAt(i);
      switch (c) {
        case '{':
          return object();
        case '[':
          return array();
        case '"':
          return string();
        case 't':
          return literal("true", Boolean.TRUE);
        case 'f':
          return literal("false", Boolean.FALSE);
        case 'n':
          return literal("null", null);
        default:
          return number();
      }
    }

    Object literal(String word, Object v) {
      if (!s.startsWith(word, i)) throw error("expected " + word);
      i += word.length();
      return v;
    }

    Map<String, Object> object() {
      Map<String, Object> m = new LinkedHashMap<>();
      i++;
      ws();
      if (s.charAt(i) == '}') {
        i++;
        return m;
      }
      while (true) {
        ws();
        String k = string();
        ws();
        if (s.charAt(i++) != ':') throw error("expected :");
        m.put(k, value());
        ws();
        char c = s.charAt(i++);
        if (c == '}') return m;
        if (c != ',') throw error("expected , or }");
      }
    }

    List<Object> array() {
      List<Object> a = new ArrayList<>();
      i++;
      ws();
      if (s.charAt(i) == ']') {
        i++;
        return a;
      }
      while (true) {
        a.add(value());
        ws();
        char c = s.charAt(i++);
        if (c == ']') return a;
        if (c != ',') throw error("expected , or ]");
      }
    }

    String string() {
      if (s.charAt(i) != '"') throw error("expected a string");
      i++;
      StringBuilder sb = new StringBuilder();
      while (true) {
        char c = s.charAt(i++);
        if (c == '"') return sb.toString();
        if (c != '\\') {
          sb.append(c);
          continue;
        }
        char e = s.charAt(i++);
        switch (e) {
          case 'n' -> sb.append('\n');
          case 'r' -> sb.append('\r');
          case 't' -> sb.append('\t');
          case 'b' -> sb.append('\b');
          case 'f' -> sb.append('\f');
          case 'u' -> {
            sb.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
            i += 4;
          }
          default -> sb.append(e);
        }
      }
    }

    Object number() {
      int start = i;
      boolean integral = true;
      while (i < s.length()) {
        char c = s.charAt(i);
        if (c == '.' || c == 'e' || c == 'E') {
          integral = false;
        } else if (!(c == '-' || c == '+' || (c >= '0' && c <= '9'))) {
          break;
        }
        i++;
      }
      String t = s.substring(start, i);
      if (t.isEmpty()) throw error("unexpected character");
      if (integral) {
        try {
          return Long.parseLong(t);
        } catch (NumberFormatException tooBig) {
          return Double.parseDouble(t);
        }
      }
      return Double.parseDouble(t);
    }
  }

  // ------------------------------------------------------------ fields

  @SuppressWarnings("unchecked")
  static Map<String, Object> obj(Object v) {
    return (Map<String, Object>) v;
  }

  @SuppressWarnings("unchecked")
  static List<Object> list(Object v) {
    return v == null ? List.of() : (List<Object>) v;
  }

  static String str(Map<String, Object> m, String k) {
    return (String) m.get(k);
  }

  static long num(Map<String, Object> m, String k) {
    Object v = m.get(k);
    return v == null ? 0 : ((Number) v).longValue();
  }

  static int integer(Map<String, Object> m, String k) {
    return (int) num(m, k);
  }

  static Float flt(Map<String, Object> m, String k) {
    Object v = m.get(k);
    return v == null ? null : ((Number) v).floatValue();
  }

  static boolean bool(Map<String, Object> m, String k) {
    return Boolean.TRUE.equals(m.get(k));
  }

  static List<String> strings(Object v) {
    List<String> out = new ArrayList<>();
    for (Object x : list(v)) out.add((String) x);
    return out;
  }

  static Map<String, Object> meta(Map<String, Object> m) {
    Object v = m.get("meta");
    return v == null ? Map.of() : obj(v);
  }
}
