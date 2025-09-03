import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl = "http://10.0.2.2:8000"; // emulator default; use deployed host for prod

  Future<Map<String, dynamic>> ask(String question) async {
    final res = await http.post(Uri.parse("$baseUrl/ask"),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"question": question}));
    if (res.statusCode != 200) {
      throw Exception("Server error ${res.statusCode}");
    }
    return jsonDecode(res.body);
  }

  Future<String> tts(String text, {String lang = "en"}) async {
    final req = http.MultipartRequest("POST", Uri.parse("$baseUrl/tts"))
      ..fields["text"] = text
      ..fields["lang"] = lang;
    final streamed = await req.send();
    final body = await streamed.stream.bytesToString();
    final data = jsonDecode(body);
    return data["audio_base64"] as String;
  }
}
